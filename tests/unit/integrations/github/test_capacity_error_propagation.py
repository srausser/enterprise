from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from integrations.github.github_manager import GithubManager
from integrations.github.github_view import GithubIssue
from integrations.models import Message, SourceType
from integrations.resolver_context import ResolverUserContext
from integrations.types import UserData
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
)
from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
    SQLAppConversationStartTaskService,
)
from openhands.app_server.errors import SandboxStartErrorCode
from openhands.app_server.sandbox.remote_sandbox_service import RemoteSandboxService
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.settings.settings_models import Settings

RETAINED_DETAIL = (
    'Retained workspace capacity (6) is exhausted; use the OpenHands UI or API '
    'to stop one explicitly selected finished sandbox, deleting its workspace, '
    'before retrying'
)
ACTIVE_DETAIL = (
    'Active runtime capacity (4) is exhausted; pause or stop a running sandbox '
    'before retrying'
)
RETAINED_MESSAGE = (
    'OpenHands retained workspace capacity is full. Use the supported OpenHands '
    'Enterprise sandbox DELETE operation on one explicitly selected finished '
    'sandbox; this permanently deletes that sandbox workspace. Pausing does not '
    'free a retained slot. Then mention @openhands again.'
)
ACTIVE_MESSAGE = (
    'OpenHands active sandbox capacity is full. Pause or stop a running sandbox, '
    'then mention @openhands again.'
)
GENERIC_MESSAGE = 'Uh oh! There was an unexpected error starting the job :('


class ReloadingStartService:
    """Reload terminal tasks before the integration consumes them."""

    def __init__(self, live_service, session_maker):
        self.live_service = live_service
        self.session_maker = session_maker
        self.reloaded_tasks = []

    async def start_app_conversation(self, request):
        async for task in self.live_service.start_app_conversation(request):
            if task.status != AppConversationStartTaskStatus.ERROR:
                yield task
                continue
            async with self.session_maker() as session:
                stored_service = SQLAppConversationStartTaskService(session)
                reloaded = await stored_service.get_app_conversation_start_task(task.id)
            assert reloaded is not None
            self.reloaded_tasks.append(reloaded)
            yield reloaded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('runtime_response', 'expected_code', 'expected_comment'),
    [
        (
            {'error': RETAINED_DETAIL},
            SandboxStartErrorCode.RETAINED_CAPACITY_EXHAUSTED,
            RETAINED_MESSAGE,
        ),
        (
            {'error': ACTIVE_DETAIL},
            SandboxStartErrorCode.ACTIVE_CAPACITY_EXHAUSTED,
            ACTIVE_MESSAGE,
        ),
        ({'error': 'internal host db-01 password=hunter2'}, None, GENERIC_MESSAGE),
        ({'error': {'unexpected': 'shape'}}, None, GENERIC_MESSAGE),
    ],
)
async def test_native_github_capacity_failure_crosses_persistence_once(
    runtime_response, expected_code, expected_comment, async_engine
):
    runtime_requests: list[httpx.Request] = []

    def runtime_handler(request: httpx.Request) -> httpx.Response:
        runtime_requests.append(request)
        if request.url.path == '/list':
            return httpx.Response(200, json={'runtimes': []})
        assert request.url.path == '/start'
        return httpx.Response(400, json=runtime_response)

    session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    settings = Settings()
    saas_user_auth = MagicMock()
    saas_user_auth.get_user_id = AsyncMock(return_value='user-id')
    saas_user_auth.get_user_email = AsyncMock(return_value=None)
    saas_user_auth.get_user_settings = AsyncMock(return_value=settings)
    user_context = ResolverUserContext(saas_user_auth)

    sandbox_spec = SandboxSpecInfo(
        id='test-image:latest',
        command=['/usr/local/bin/openhands-agent-server', '--port', '60000'],
        initial_env={},
        working_dir='/workspace/project',
    )
    sandbox_spec_service = AsyncMock()
    sandbox_spec_service.get_default_sandbox_spec.return_value = sandbox_spec

    comments: list[str] = []
    async with (
        session_maker() as session,
        httpx.AsyncClient(
            transport=httpx.MockTransport(runtime_handler)
        ) as runtime_client,
    ):
        task_service = SQLAppConversationStartTaskService(session)
        sandbox_service = RemoteSandboxService(
            sandbox_spec_service=sandbox_spec_service,
            api_url='https://runtime.invalid',
            api_key='recording-runtime-key',
            web_url=None,
            resource_factor=1,
            runtime_class=None,
            start_sandbox_timeout=30,
            max_num_sandboxes=10,
            user_context=user_context,
            httpx_client=runtime_client,
            db_session=session,
        )
        live_service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=False,
            user_context=user_context,
            app_conversation_info_service=MagicMock(),
            app_conversation_start_task_service=task_service,
            event_callback_service=MagicMock(),
            event_service=MagicMock(),
            sandbox_service=sandbox_service,
            sandbox_spec_service=sandbox_spec_service,
            jwt_service=MagicMock(),
            pending_message_service=MagicMock(),
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=0,
            max_num_conversations_per_sandbox=20,
            httpx_client=runtime_client,
            web_url=None,
            openhands_provider_base_url=None,
            access_token_hard_timeout=None,
            app_mode='test',
        )
        live_service._reserve_daily_conversation_quota = AsyncMock(return_value=False)
        reloading_service = ReloadingStartService(live_service, session_maker)

        @asynccontextmanager
        async def get_service(_injector_state):
            yield reloading_service

        github_view = GithubIssue(
            issue_number=1690,
            installation_id=1,
            full_repo_name='RausserHQ/homelab-platform',
            is_public_repo=False,
            user_info=UserData(
                user_id=123,
                username='testuser',
                keycloak_user_id='keycloak-id',
            ),
            raw_payload=Message(
                source=SourceType.GITHUB,
                message={'payload': {'issue': {'number': 1690}}},
            ),
            conversation_id='',
            uuid='trigger-id',
            should_extract=True,
            send_summary_instruction=False,
            title='Capacity propagation',
            description='Test issue',
            previous_comments=[],
        )
        github_view._get_v1_initial_user_message = AsyncMock(return_value='Fix it')

        token_manager = MagicMock()
        token_manager.get_idp_token_from_idp_user_id = AsyncMock(
            return_value='recording-provider-token'
        )
        data_collector = MagicMock()
        data_collector.save_data = AsyncMock()

        async def record_comment(message, _view):
            comments.append(message)

        with (
            patch('integrations.github.github_manager.Auth'),
            patch('integrations.github.github_manager.GithubIntegration'),
            patch(
                'integrations.github.github_manager.get_saas_user_auth',
                new=AsyncMock(return_value=saas_user_auth),
            ),
            patch(
                'integrations.github.github_view.resolve_org_for_repo',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'integrations.github.github_view.get_app_conversation_service',
                new=get_service,
            ),
        ):
            manager = GithubManager(token_manager, data_collector)
            manager.send_message = record_comment
            await manager.start_job(github_view)

    assert [request.url.path for request in runtime_requests].count('/start') == 1
    assert comments == [expected_comment]
    assert "I'm on it!" not in comments[0]
    assert 'hunter2' not in comments[0]
    assert len(reloading_service.reloaded_tasks) == 1
    assert reloading_service.reloaded_tasks[0].error_code == expected_code
