"""Repository hooks must finish successfully before conversation startup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
)
from openhands.sdk.workspace.models import CommandResult


class ShellWorkspace:
    """Execute the real command through a shell, without a remote sandbox."""

    async def execute_command(self, command, cwd, timeout):
        process = await asyncio.create_subprocess_exec(
            '/bin/sh',
            '-c',
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return CommandResult(
            command=command,
            exit_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            timeout_occurred=False,
        )


@pytest.mark.asyncio
async def test_setup_runs_bash_without_executable_mode_and_quotes_path(tmp_path):
    project = tmp_path / "repo's $(touch injected) directory"
    hook = project / '.openhands/setup.sh'
    hook.parent.mkdir(parents=True)
    hook.write_text('[[ "$BASH_VERSION" ]] && printf ready > setup-ready\n')
    hook.chmod(0o600)

    await AppConversationServiceBase.maybe_run_setup_script(
        None, ShellWorkspace(), str(project)
    )

    assert (project / 'setup-ready').read_text() == 'ready'
    assert not (project / 'injected').exists()
    assert hook.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_missing_setup_hook_is_noop(tmp_path):
    await AppConversationServiceBase.maybe_run_setup_script(
        None, ShellWorkspace(), str(tmp_path)
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_kind', ['nonzero', 'directory', 'broken_symlink'])
async def test_present_invalid_or_failing_setup_stops_startup(tmp_path, hook_kind):
    hook = tmp_path / '.openhands/setup.sh'
    hook.parent.mkdir()
    if hook_kind == 'nonzero':
        hook.write_text('echo synthetic-secret >&2; exit 23\n')
    elif hook_kind == 'directory':
        hook.mkdir()
    else:
        hook.symlink_to('missing-target')

    with pytest.raises(RuntimeError, match='Repository setup .*failed') as error:
        await AppConversationServiceBase.maybe_run_setup_script(
            None, ShellWorkspace(), str(tmp_path)
        )
    assert 'synthetic-secret' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exit_code,timed_out', [(23, False), (-1, True), (0, True), (-1, False)]
)
async def test_setup_result_fails_closed_without_exposing_output(exit_code, timed_out):
    workspace = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=CommandResult(
                command='setup',
                exit_code=exit_code,
                stdout='synthetic-secret',
                stderr='synthetic-secret',
                timeout_occurred=timed_out,
            )
        )
    )
    with pytest.raises(RuntimeError) as error:
        await AppConversationServiceBase.maybe_run_setup_script(
            None, workspace, '/project'
        )
    assert 'synthetic-secret' not in str(error.value)
    assert (
        'timed out after 600 seconds' in str(error.value)
        if timed_out
        else f'exit code {exit_code}' in str(error.value)
    )
    assert workspace.execute_command.call_args.kwargs['timeout'] == 600


@pytest.mark.asyncio
async def test_setup_failure_prevents_subsequent_startup_stages(tmp_path):
    hook = tmp_path / '.openhands/setup.sh'
    hook.parent.mkdir()
    hook.write_text('exit 23\n')
    workspace = ShellWorkspace()
    workspace.working_dir = str(tmp_path)
    service = SimpleNamespace(
        clone_or_init_git_repo=AsyncMock(),
        maybe_run_setup_script=lambda workspace,
        project_dir: AppConversationServiceBase.maybe_run_setup_script(
            None, workspace, project_dir
        ),
        maybe_setup_git_hooks=AsyncMock(),
        load_and_merge_all_skills=AsyncMock(),
    )
    task = SimpleNamespace(
        status=None, request=SimpleNamespace(selected_repository=None)
    )
    statuses = []
    with pytest.raises(RuntimeError, match='exit code 23'):
        async for update in AppConversationServiceBase.run_setup_scripts(
            service, task, None, workspace, '', None
        ):
            statuses.append(update.status)
    assert statuses == [
        AppConversationStartTaskStatus.PREPARING_REPOSITORY,
        AppConversationStartTaskStatus.RUNNING_SETUP_SCRIPT,
    ]
    service.maybe_setup_git_hooks.assert_not_called()
    service.load_and_merge_all_skills.assert_not_called()
