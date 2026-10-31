"""Wraps vLLM's async engine running Orpheus 3B and streams generated audio token ids per request.

One `TTSEngine` owns one vLLM `AsyncLLM` (vLLM 0.32's async engine; the model runs in a separate
engine-core process). Every `stream()` call becomes its own vLLM request with a unique id, and vLLM's
scheduler batches all active requests into the same decode steps, so a single instance serves many
concurrent users (the step 3 server shares one).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from . import config as C


def build_prompt_ids(tokenizer, text: str, voice: str) -> list[int]:
    """Orpheus prompt as token ids (upstream engine_class.py `_format_prompt`).

    Passed straight to vLLM as ids; upstream decodes to a string and lets vLLM re-tokenize it.
    """
    text_ids = tokenizer(f"{voice}: {text}").input_ids  # includes Llama BOS 128000, as upstream
    return [C.START_OF_HUMAN, *text_ids, C.END_OF_TEXT, C.END_OF_HUMAN, C.START_OF_AI, C.START_OF_SPEECH]


class TTSEngine:
    def __init__(self, preset: str = C.DEFAULT_PRESET, sampling: C.SamplingConfig | None = None,
                 **engine_overrides):
        self.preset = preset
        self.engine_args = {**C.engine_args(preset), **engine_overrides}
        self.sampling = sampling or C.SamplingConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(C.TOKENIZER)
        assert self.tokenizer.convert_tokens_to_ids("<custom_token_10>") == C.AUDIO_TOKEN_BASE
        self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.engine_args))

    def sampling_params(self, **overrides) -> SamplingParams:
        s = {**self.sampling.to_dict(), **overrides}
        # DELTA: each output carries only the tokens generated since the previous one.
        # detokenize=False: the text form of audio tokens is never used (step 1d).
        return SamplingParams(**s, detokenize=False, output_kind=RequestOutputKind.DELTA)

    async def stream(self, text: str, voice: str = C.DEFAULT_VOICE, request_id: str | None = None,
                     **sampling_overrides) -> AsyncIterator[list[int]]:
        """Yield lists of newly generated token ids (raw vLLM ids, incl. the final stop token).

        If the consumer stops iterating early (e.g. a client disconnects), the request is aborted so it
        stops using GPU time.
        """
        request_id = request_id or uuid.uuid4().hex
        prompt = TokensPrompt(prompt_token_ids=build_prompt_ids(self.tokenizer, text, voice))
        finished = False
        try:
            async for out in self.llm.generate(prompt, self.sampling_params(**sampling_overrides), request_id):
                comp = out.outputs[0]
                if comp.token_ids:
                    yield list(comp.token_ids)
                if out.finished:
                    finished = True
        finally:
            if not finished:
                await self.llm.abort(request_id)

    def kv_cache_tokens(self) -> int | None:
        cache = self.llm.vllm_config.cache_config
        return cache.num_gpu_blocks * cache.block_size if cache.num_gpu_blocks else None

    def shutdown(self) -> None:
        self.llm.shutdown()
