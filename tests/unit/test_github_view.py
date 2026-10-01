from unittest import TestCase, mock
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from integrations.github.github_view import (
    GithubFactory,
    GithubInvocationError,
    GithubIssue,
    GithubIssueComment,
    _parse_invocation_overrides,
    _resolve_model_override,
    get_oh_labels,
)
from integrations.models import Message, SourceType
from integrations.types import UserData
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
)
from openhands.app_server.errors import SandboxStartError, SandboxStartErrorCode
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.utils.llm import ModelsResponse
from openhands.sdk.settings import ACPAgentSettings


class TestGithubLabels(TestCase):
    def test_labels_with_staging(self):
        oh_label, inline_oh_label = get_oh_labels('staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_staging_v2(self):
        oh_label, inline_oh_label = get_oh_labels('main.staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_local(self):
        oh_label, inline_oh_label = get_oh_labels('localhost:3000')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_prod(self):
        oh_label, inline_oh_label = get_oh_labels('app.all-hands.dev')
        self.assertEqual(oh_label, 'openhands')
        self.assertEqual(inline_oh_label, '@openhands')

    def test_labels_with_spaces(self):
        """Test that spaces are properly stripped"""
        oh_label, inline_oh_label = get_oh_labels('  local  ')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    @mock.patch.dict('os.environ', {'OH_RESOLVER_LABEL': 'openhands-dev'})
    def test_labels_with_override(self):
        """An explicit OH_RESOLVER_LABEL overrides host inference."""
        oh_label, inline_oh_label = get_oh_labels('app.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-dev')
        self.assertEqual(inline_oh_label, '@openhands-dev')

    @mock.patch.dict('os.environ', {'OH_RESOLVER_LABEL': '  openhands-dev  '})
    def test_labels_override_is_stripped(self):
        """The override is trimmed before use."""
        oh_label, inline_oh_label = get_oh_labels('staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-dev')
        self.assertEqual(inline_oh_label, '@openhands-dev')

    @mock.patch.dict('os.environ', {'OH_RESOLVER_LABEL': ''})
    def test_labels_empty_override_falls_back(self):
        """An empty override falls back to host inference."""
        oh_label, inline_oh_label = get_oh_labels('staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')


class TestGithubCommentTriggers:
    COMMENT_CLASSIFIERS = (
        (GithubFactory.is_issue_comment, {'issue': {'number': 1}}),
        (
            GithubFactory.is_pr_comment,
            {'issue': {'number': 1, 'pull_request': {}}},
        ),
        (GithubFactory.is_inline_pr_comment, {'pull_request': {}}),
    )

    @pytest.mark.parametrize(
        ('body', 'expected'),
        [
            ('@openhands please help', True),
            ('\n  @OPENHANDS please help', True),
            ('\t@OpenHands please help', True),
            ('Use `@openhands please help` to invoke the resolver.', False),
            ('> @openhands please help', False),
        ],
    )
    def test_comment_trigger_requires_leading_mention(self, body, expected):
        with mock.patch(
            'integrations.github.github_view.INLINE_OH_LABEL', '@openhands'
        ):
            for classifier, payload in self.COMMENT_CLASSIFIERS:
                message = Message(
                    source=SourceType.GITHUB,
                    message={
                        'payload': {
                            'action': 'created',
                            'comment': {'body': body},
                            **payload,
                        }
                    },
                )

                assert classifier(message) is expected


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


class TestGithubV1ConversationRouting(TestCase):
    """Test V1 conversation routing logic in GitHub integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a proper UserData instance instead of MagicMock
        self.user_data = UserData(
            user_id=123, username='testuser', keycloak_user_id='test-keycloak-id'
        )

        # Create a mock raw_payload
        self.raw_payload = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'opened',
                    'issue': {'number': 123},
                }
            },
        )

    def _create_github_issue(self):
        """Create a GithubIssue instance for testing."""
        return GithubIssue(
            user_info=self.user_data,
            full_repo_name='test/repo',
            issue_number=123,
            installation_id=456,
            conversation_id='test-conversation-id',
            should_extract=True,
            send_summary_instruction=False,
            is_public_repo=True,
            raw_payload=self.raw_payload,
            uuid='test-uuid',
            title='Test Issue',
            description='Test issue description',
            previous_comments=[],
        )

    def _create_github_issue_comment(self, comment_body):
        issue = self._create_github_issue()
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
    @patch.object(GithubIssue, '_create_v1_conversation')
    async def test_create_new_conversation_routes_to_v1(self, mock_create_v1):
        """Test that conversation creation routes to V1."""
        mock_create_v1.return_value = None

        github_issue = self._create_github_issue()

        # Mock parameters
        jinja_env = MagicMock()
        git_provider_tokens = MagicMock()
        conversation_metadata = MagicMock()
        saas_user_auth = MagicMock()

        # Call the method
        await github_issue.create_new_conversation(
            jinja_env, git_provider_tokens, conversation_metadata, saas_user_auth
        )

        # Verify V1 was called
        mock_create_v1.assert_called_once_with(
            jinja_env, saas_user_auth, conversation_metadata
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

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    async def test_create_v1_conversation_restores_classified_task_error(
        self, mock_get_service
    ):
        github_issue = self._create_github_issue()
        github_issue._get_v1_initial_user_message = AsyncMock(return_value='Fix it')
        failed_task = AppConversationStartTask(
            created_by_user_id='user-id',
            status=AppConversationStartTaskStatus.ERROR,
            detail='Failed to start sandbox',
            error_code=SandboxStartErrorCode.ACTIVE_CAPACITY_EXHAUSTED,
            request=AppConversationStartRequest(),
        )

        service = MagicMock()

        async def start_app_conversation(_request):
            yield failed_task

        service.start_app_conversation = start_app_conversation
        mock_get_service.return_value.__aenter__ = AsyncMock(return_value=service)
        mock_get_service.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(SandboxStartError) as exc_info:
            await github_issue._create_v1_conversation(
                MagicMock(), MagicMock(), UUID(int=1)
            )

        assert (
            exc_info.value.error_code == SandboxStartErrorCode.ACTIVE_CAPACITY_EXHAUSTED
        )


class TestGithubOrgRouting(TestCase):
    """Test org routing for GitHub resolver conversations."""

    def setUp(self):
        self.user_data = UserData(
            user_id=123, username='testuser', keycloak_user_id='test-keycloak-id'
        )
        self.raw_payload = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'opened',
                    'issue': {'number': 42},
                }
            },
        )
        self.resolved_org_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    def _create_github_issue(self):
        return GithubIssue(
            user_info=self.user_data,
            full_repo_name='ClaimedOrg/repo',
            issue_number=42,
            installation_id=456,
            conversation_id='',
            should_extract=True,
            send_summary_instruction=False,
            is_public_repo=True,
            raw_payload=self.raw_payload,
            uuid='test-uuid',
            title='',
            description='',
            previous_comments=[],
        )

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    @patch('integrations.github.github_view.resolve_org_for_repo')
    async def test_v1_passes_resolver_org_id_to_resolver_user_context(
        self, mock_resolve_org, mock_get_service
    ):
        """V1 path passes resolved org_id to ResolverUserContext."""
        # Arrange
        mock_resolve_org.return_value = self.resolved_org_id

        github_issue = self._create_github_issue()

        # Initialize to set resolved_org_id
        await github_issue.initialize_new_conversation()

        # Assert
        assert github_issue.resolved_org_id == self.resolved_org_id

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    @patch('integrations.github.github_view.resolve_org_for_repo')
    async def test_no_claim_passes_none_resolver_org_id(
        self, mock_resolve_org, mock_get_service
    ):
        """When no claim exists, resolver_org_id is None (falls back to personal workspace)."""
        # Arrange
        mock_resolve_org.return_value = None

        github_issue = self._create_github_issue()

        # Act
        await github_issue.initialize_new_conversation()

        # Assert
        assert github_issue.resolved_org_id is None
