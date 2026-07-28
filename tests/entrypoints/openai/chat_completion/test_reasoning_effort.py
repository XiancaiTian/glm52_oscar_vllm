# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)


def test_reasoning_effort_max_is_accepted():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "test"}],
        reasoning_effort="max",
    )

    assert request.reasoning_effort == "max"
    assert (
        request.build_chat_params(None, "auto").chat_template_kwargs["reasoning_effort"]
        == "max"
    )
