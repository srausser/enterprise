from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from integrations.github.github_view import (
    GithubInvocationError,
    GithubIssueComment,
    _parse_invocation_overrides,
    _resolve_model_override,
)
from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
)
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.utils.llm import ModelsResponse
from openhands.sdk.settings import ACPAgentSettings
from tests.unit.test_github_view import (
    TestGithubV1ConversationRouting as _RoutingFixtures,
)


class TestGithubInvocationOverrides:
    @pytest.mark.parametrize(
        ('body', 'expected_body', 'model', 'effort'),
        [
            (
                '@openhands model=gpt-5.6-sol effort=xhigh\nFix the failing CI job.',
                '@openhands\nFix the failing CI job.',
                'gpt-5.6-sol',
                'xhigh',
            ),
            (
                '@OpenHands model=openai/gpt-5 Fix this.',
                '@OpenHands Fix this.',
                'openai/gpt-5',
                None,
            ),
            (
                'Please @openhands effort=low investigate.',
                'Please @openhands effort=low investigate.',
                None,
                None,
            ),
            (
                '@openhands Fix this.',
                '@openhands Fix this.',
                None,
                None,
            ),
        ],
    )
    def test_parse_invocation_overrides(self, body, expected_body, model, effort):
        assert _parse_invocation_overrides(body) == (
            expected_body,
            model,
            effort,
        )

    @pytest.mark.parametrize(
        'body',
        [
            '@openhands model= Fix this.',
            '@openhands effort=extreme Fix this.',
            '@openhands effort=low effort=high Fix this.',
        ],
    )
    def test_invalid_invocation_overrides_are_actionable(self, body):
        with pytest.raises(
            GithubInvocationError, match=r'Invalid @openhands invocation'
        ):
            _parse_invocation_overrides(body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('invocation_model', 'expected_model'),
        [
            ('gpt-5.6-sol', 'openhands/gpt-5.6-sol'),
            ('openhands/gpt-5.6-sol', 'openhands/gpt-5.6-sol'),
            ('openai/gpt-5.6-sol', 'openai/gpt-5.6-sol'),
        ],
    )
    @patch('integrations.github.github_view.get_global_config')
    async def test_model_resolves_using_real_service_contract(
        self, mock_get_global_config, invocation_model, expected_model
    ):
        model_service = DefaultLLMModelService()
        model_service._cached_response = ModelsResponse(
            models=['openhands/gpt-5.6-sol', 'openai/gpt-5.6-sol'],
            verified_models=['gpt-5.6-sol'],
            verified_providers=['openhands'],
            default_model='openhands/gpt-5.6-sol',
        )
        injector = MagicMock()
        injector.context.return_value.__aenter__ = AsyncMock(return_value=model_service)
        injector.context.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_global_config.return_value.llm_model = injector

        if invocation_model == 'gpt-5.6-sol':
            cleaned, invocation_model, effort = _parse_invocation_overrides(
                '@openhands model=gpt-5.6-sol effort=xhigh Fix the failing CI job.'
            )
            assert cleaned == '@openhands Fix the failing CI job.'
            assert effort == 'xhigh'
        assert await _resolve_model_override(invocation_model) == expected_model

        page = await model_service.search_llm_models(limit=100)
        assert ('openhands', 'gpt-5.6-sol') in {
            (item.provider, item.name) for item in page.items
        }

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_global_config')
    async def test_unavailable_model_is_actionable(self, mock_get_global_config):
        model_service = AsyncMock()
        model_service.search_llm_models = AsyncMock(return_value=MagicMock(items=[]))
        injector = MagicMock()
        injector.context.return_value.__aenter__ = AsyncMock(return_value=model_service)
        injector.context.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_global_config.return_value.llm_model = injector

        with pytest.raises(GithubInvocationError, match='model .* is not available'):
            await _resolve_model_override('missing-model')

    def _create_github_issue_comment(self, comment_body):
        fixtures = _RoutingFixtures()
        fixtures.setUp()
        issue = fixtures._create_github_issue()
        return GithubIssueComment(
            **issue.__dict__,
            comment_body=comment_body,
            comment_id=789,
        )

    def test_trigger_comment_history_uses_cleaned_body(self):
        github_issue = self._create_github_issue_comment(
            '@openhands model=gpt-5.6-sol Fix it.'
        )
        github_issue.comment_body = '@openhands Fix it.'
        trigger_comment = MagicMock(id='789')
        cleaned_comment = MagicMock()
        trigger_comment.model_copy.return_value = cleaned_comment
        prior_comment = MagicMock(id='123')
        github_issue.previous_comments = [prior_comment, trigger_comment]

        github_issue._clean_trigger_comment_history()

        assert github_issue.previous_comments == [prior_comment, cleaned_comment]
        trigger_comment.model_copy.assert_called_once_with(
            update={'body': '@openhands Fix it.'}
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('comment_body', 'expected_text', 'expected_model', 'expected_effort'),
        [
            (
                '@openhands model=gpt-5.6-sol effort=xhigh\nFix it.',
                '@openhands\nFix it.',
                'openhands/gpt-5.6-sol',
                'xhigh',
            ),
            ('@openhands Fix it.', '@openhands Fix it.', None, None),
        ],
    )
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_comment_overrides_reach_start_request_without_task_tokens(
        self,
        mock_get_service,
        comment_body,
        expected_text,
        expected_model,
        expected_effort,
    ):
        github_issue = self._create_github_issue_comment(comment_body)
        github_issue.resolved_org_id = None

        async def initial_message(_jinja_env):
            return github_issue.comment_body

        github_issue._get_v1_initial_user_message = initial_message
        service = MagicMock()
        requests = []

        async def start_app_conversation(request):
            requests.append(request)
            if False:
                yield

        service.start_app_conversation = start_app_conversation
        mock_get_service.return_value.__aenter__ = AsyncMock(return_value=service)
        mock_get_service.return_value.__aexit__ = AsyncMock(return_value=False)
        saas_user_auth = MagicMock()
        saas_user_auth.get_user_settings = AsyncMock(return_value=Settings())
        saas_user_auth.get_user_id = AsyncMock(return_value='user-id')

        with patch(
            'integrations.github.github_view._resolve_model_override',
            new=AsyncMock(return_value='openhands/gpt-5.6-sol'),
        ):
            await github_issue._create_v1_conversation(
                MagicMock(), saas_user_auth, UUID(int=1)
            )

        assert len(requests) == 1
        request = requests[0]
        assert request.llm_model == expected_model
        assert request.reasoning_effort == expected_effort
        assert request.initial_message.content[0].text == expected_text

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_acp_override_fails_before_conversation_service(
        self, mock_get_service
    ):
        github_issue = self._create_github_issue_comment(
            '@openhands effort=xhigh Fix it.'
        )
        github_issue.resolved_org_id = None
        saas_user_auth = MagicMock()
        saas_user_auth.get_user_settings = AsyncMock(
            return_value=Settings(agent_settings=ACPAgentSettings())
        )
        saas_user_auth.get_user_id = AsyncMock(return_value='user-id')

        with pytest.raises(GithubInvocationError, match='native OpenHands agent'):
            await github_issue._create_v1_conversation(
                MagicMock(), saas_user_auth, UUID(int=1)
            )

        mock_get_service.assert_not_called()

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_invalid_override_fails_before_conversation_service(
        self, mock_get_service
    ):
        github_issue = self._create_github_issue_comment(
            '@openhands effort=extreme Fix it.'
        )

        with pytest.raises(GithubInvocationError, match='effort must be one of'):
            await github_issue._create_v1_conversation(
                MagicMock(), MagicMock(), UUID(int=1)
            )

        mock_get_service.assert_not_called()

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_unavailable_model_fails_before_conversation_service(
        self, mock_get_service
    ):
        github_issue = self._create_github_issue_comment(
            '@openhands model=missing-model Fix it.'
        )
        github_issue.resolved_org_id = None
        saas_user_auth = MagicMock()
        saas_user_auth.get_user_settings = AsyncMock(return_value=Settings())
        saas_user_auth.get_user_id = AsyncMock(return_value='user-id')

        with (
            patch(
                'integrations.github.github_view._resolve_model_override',
                new=AsyncMock(
                    side_effect=GithubInvocationError(
                        "Invalid @openhands invocation: model 'missing-model' is not available."
                    )
                ),
            ),
            pytest.raises(GithubInvocationError, match='model .* is not available'),
        ):
            await github_issue._create_v1_conversation(
                MagicMock(), saas_user_auth, UUID(int=1)
            )

        mock_get_service.assert_not_called()
