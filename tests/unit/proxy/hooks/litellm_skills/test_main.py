import os
import subprocess
import sys
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

SKILL_TOOL_NAME = "litellm_skill_e2b8dca8_031a_4481_b034_b9ec7d4eb7bf"


def _request_data():
    return {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "run the skill"}],
        "litellm_metadata": {
            "_litellm_code_execution_enabled": True,
            "_skill_files": {SKILL_TOOL_NAME: {"main.py": b"print('hi')"}},
        },
    }


def _tool_use_response(tool_name):
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}}
        ],
    }


@pytest.mark.asyncio
async def test_post_call_success_hook_executes_litellm_skill_tool():
    """DB skill tool names carry the litellm_skill_ prefix and must trigger the execution loop."""
    hook = SkillsInjectionHook()
    response = _tool_use_response(SKILL_TOOL_NAME)

    with patch.object(
        hook, "_execute_code_loop_messages_api", new=AsyncMock(return_value=response)
    ) as mock_loop:
        result = await hook.async_post_call_success_deployment_hook(
            request_data=_request_data(), response=response, call_type=None
        )

    mock_loop.assert_awaited_once()
    assert result is response


@pytest.mark.asyncio
async def test_execute_code_loop_dispatches_litellm_skill_tool():
    """The agentic loop must route litellm_skill_ tool calls to _execute_skill_tool."""
    hook = SkillsInjectionHook()
    final_response = {"stop_reason": "end_turn", "content": []}

    with (
        patch.object(
            hook, "_execute_skill_tool", new=AsyncMock(return_value="skill ran")
        ) as mock_exec,
        patch("litellm.anthropic.acreate", new=AsyncMock(return_value=final_response)),
    ):
        result = await hook._execute_code_loop_messages_api(
            data=_request_data(),
            response=_tool_use_response(SKILL_TOOL_NAME),
            skill_files={"main.py": b"print('hi')"},
        )

    mock_exec.assert_awaited_once()
    assert mock_exec.await_args.args[0] == SKILL_TOOL_NAME
    assert result is final_response


def test_importing_proxy_hooks_does_not_mutate_litellm_callbacks():
    script: Final = (
        "import litellm; "
        "litellm.callbacks = []; "
        "import litellm.proxy.hooks; "
        "assert len(litellm.callbacks) == 0, f'mutated: {litellm.callbacks}'"
    )
    result: Final = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_is_importable_skill_path_accepts_identifier_paths():
    from litellm.proxy.hooks.litellm_skills.main import _is_importable_skill_path

    assert _is_importable_skill_path("main.py")
    assert _is_importable_skill_path("core/gif_builder.py")
    assert _is_importable_skill_path("_private/module_2.py")


def test_is_importable_skill_path_rejects_injection_payloads():
    from litellm.proxy.hooks.litellm_skills.main import _is_importable_skill_path

    malicious_paths = [
        "x'+__import__('os').system('id')+'.py",
        "a/__import__('os').py",
        "a b/main.py",
        "../escape.py",
        "dir/../main.py",
        "a;b/main.py",
        "main .py",
        "loop.py\x00",
    ]
    for path in malicious_paths:
        assert not _is_importable_skill_path(path), path


@pytest.mark.asyncio
async def test_execute_skill_tool_rejects_malicious_module_path():
    """A skill filename carrying code-injection payloads must never reach the generated code."""
    hook = SkillsInjectionHook()
    malicious = "x'+__import__('os').system('touch /tmp/pwned')+'.py"
    with patch.object(
        hook,
        "_execute_code",
        new=AsyncMock(return_value="executed"),
    ) as mock_exec:
        await hook._execute_skill_tool(
            tool_name=SKILL_TOOL_NAME,
            tool_input={},
            skill_files={"SKILL.md": b"x", malicious: b"print('hi')"},
            executor=object(),
            generated_files=[],
        )
    generated_code = mock_exec.await_args.args[0]
    assert malicious not in generated_code
    assert "__import__('os')" not in generated_code
    assert "No executable skill module found" in generated_code
