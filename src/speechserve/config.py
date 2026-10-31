"""Central configuration: model presets, vLLM engine flags, sampling, Orpheus token layout, and chunking.

Everything here was established in step 1 (scripts/step1_generate.py, docs/DECISIONS.md); this module is
the single place the serving code reads it from. Nothing in this file imports torch or vllm, so it is
safe to import in CPU-only tests.

Upstream source of truth: canopyai/Orpheus-TTS @ e64661fe6d.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# ---- Models ---------------------------------------------------------------------------------------
BASE_MODEL = "canopylabs/orpheus-3b-0.1-ft"
# Upstream engine_class.py loads the tokenizer from the pretrained repo (same Llama-3 tokenizer).
TOKENIZER = "canopylabs/orpheus-3b-0.1-pretrained"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
# Directory holding the quantized checkpoints (orpheus-3b-w4a16, orpheus-3b-w4a16-lmhead). Colab default is
# Drive; Docker and other machines set SPEECHSERVE_MODELS_DIR.
MODELS_DIR = os.environ.get("SPEECHSERVE_MODELS_DIR", "/content/drive/MyDrive/speech-infra/models")

# Engine flags shared by every preset (DECISIONS.md "Model and serving stack").
BASE_ENGINE_ARGS = dict(
    tokenizer=TOKENIZER,
    dtype="bfloat16",
    max_model_len=4096,            # 131k advertised context does not fit next to the weights on an L4
    gpu_memory_utilization=0.85,   # leaves room for SNAC on the same GPU
)

# name -> engine-arg overrides. ms/token = single-stream decode speed measured in step 1b-1d.
PRESETS: dict[str, dict] = {
    "bf16": {"model": BASE_MODEL},                                          # 27.4 ms/token
    "w4a16": {"model": f"{MODELS_DIR}/orpheus-3b-w4a16"},                   # 10.9 ms/token
    "w4a16_lmhead": {"model": f"{MODELS_DIR}/orpheus-3b-w4a16-lmhead"},     # 8.1 ms/token
}
DEFAULT_PRESET = "w4a16_lmhead"


def engine_args(preset: str = DEFAULT_PRESET) -> dict:
    return {**BASE_ENGINE_ARGS, **PRESETS[preset]}


# ---- Orpheus prompt / token layout (engine_class.py `_format_prompt`, decoder.py) -------------------
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
START_OF_AI = 128261
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258       # stop token (realtime_streaming_example/main.py)
AUDIO_TOKEN_BASE = 128266    # id of <custom_token_10>: code 0 at frame position 0
CODEBOOK_SIZE = 4096
TOKENS_PER_FRAME = 7
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 2048     # one 7-token frame decodes to 2048 samples (~85.3 ms)

DEFAULT_VOICE = "tara"


# ---- Sampling ---------------------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    """Upstream `generate_tokens_sync` defaults, with the corrected stop token."""
    temperature: float = 0.6
    top_p: float = 0.8
    repetition_penalty: float = 1.3   # >= 1.1 required; at 1.0 generation never stopped (step 1d)
    # Upstream uses 1200 (~14.6 s of audio). Raised in step 2 so ~20-30 s paragraphs are not cut off;
    # prompt + 3000 still fits in max_model_len=4096.
    max_tokens: int = 3000
    stop_token_ids: list[int] = field(default_factory=lambda: [END_OF_SPEECH])
    seed: int | None = 0              # fixed for reproducible measurements; None = random per request

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Streaming chunks -------------------------------------------------------------------------------
@dataclass
class ChunkConfig:
    """How the token stream is cut into decode calls (see codec.py for the full explanation).

    The defaults reproduce upstream decoder.py `tokens_decoder`: decode the last 4 frames each time a
    new frame arrives and keep the middle samples [2048:4096], i.e. 1 frame of left context and 2 frames
    of right context around the 1 frame that is emitted.
    """
    first_chunk_frames: int = 1   # audio frames in the first chunk (tuned in step 6)
    # Frames the first chunk waits for (step 6c). 4 = upstream: decode frames 0-3, emit frame 1 (frame 0 is
    # never played). W < 4: decode frames 0..W-1 and emit frame 0 with W-1 frames of right context; the next
    # frames may also go out with fewer right frames (never fewer than W-1) until the client's buffer allows
    # waiting for the standard 2 (see codec.ChunkPlanner).
    first_wait_frames: int = 4
    # Max audio frames in each later chunk. 1 = upstream (one window per frame). > 1 (step 6b): chunks
    # grow in powers of two up to this cap, but only while the client's buffer can cover the wait
    # (see codec.ChunkPlanner); without timing info the planner waits for full-size chunks.
    chunk_frames: int = 1
    chunk_safety_ms: float = 50.0  # extra margin on top of the measured SNAC time when deciding to wait
    left_context: int = 1         # frames decoded before the emitted ones and thrown away (upstream: 1)
    right_context: int = 2        # frames decoded after the emitted ones and thrown away (upstream: 2)
    skip_first_frame: bool = True  # upstream never emits frame 0 (its window starts at frame 1)
    flush_tail: bool = True       # emit the last right_context frames at end of stream (upstream drops them)

    def to_dict(self) -> dict:
        return asdict(self)
