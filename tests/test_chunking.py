"""CPU-only tests for the token -> frame -> chunk-window logic in speechserve.codec (no GPU, no SNAC)."""

import numpy as np
import pytest

from speechserve import config as C
from speechserve.codec import ChunkPlanner, FrameAssembler, Window, frames_to_layers, token_to_code


def audio_ids(codes):
    """Inverse mapping: codes (one per audio position) -> Orpheus token ids."""
    return [C.AUDIO_TOKEN_BASE + c + (i % 7) * C.CODEBOOK_SIZE for i, c in enumerate(codes)]


def test_token_to_code_roundtrip_and_range():
    codes = list(range(14))
    ids = audio_ids(codes)
    assert [token_to_code(t, i) for i, t in enumerate(ids)] == codes
    assert token_to_code(C.AUDIO_TOKEN_BASE + 6 * 4096 + 4095, 6) == 4095
    with pytest.raises(ValueError):
        token_to_code(C.AUDIO_TOKEN_BASE + 4096, 0)   # position-1 range used at position 0


def test_assembler_skips_non_audio_and_splits_frames():
    asm = FrameAssembler()
    ids = audio_ids(range(17))
    # deliver in odd-sized pieces, with the stop token at the end
    assert asm.push(ids[:3]) == 0
    assert asm.push(ids[3:10]) == 1
    assert asm.push(ids[10:] + [C.END_OF_SPEECH]) == 2
    assert asm.frames == [list(range(7)), list(range(7, 14))]
    assert asm.n_audio_tokens == 17 and asm.n_tokens == 18


def test_frames_to_layers_matches_upstream_layout():
    l0, l1, l2 = frames_to_layers([[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16]])
    assert l0.tolist() == [0, 10]
    assert l1.tolist() == [1, 4, 11, 14]
    assert l2.tolist() == [2, 3, 5, 6, 12, 13, 15, 16]


def test_default_plan_is_upstream_sliding_window():
    """Upstream: at 28, 35, 42... audio tokens decode the last 4 frames and keep samples [2048:4096]."""
    p = ChunkPlanner(C.ChunkConfig(flush_tail=False))
    assert p.plan(3) == []
    assert p.plan(4) == [Window(0, 4, 1, 2)]
    assert p.plan(4) == []
    assert p.plan(5) == [Window(1, 5, 2, 3)]
    assert Window(0, 4, 1, 2).keep_samples == slice(2048, 4096)
    assert p.plan(5, final=True) == []        # upstream drops the last 2 frames


def test_flush_tail_emits_remaining_frames():
    p = ChunkPlanner(C.ChunkConfig())
    p.plan(6)
    assert p.plan(6, final=True) == [Window(3, 6, 4, 6)]


def test_short_stream_flushes_everything():
    p = ChunkPlanner(C.ChunkConfig(first_chunk_frames=4))
    assert p.plan(3) == []
    assert p.plan(3, final=True) == [Window(0, 3, 1, 3)]


@pytest.mark.parametrize("first,chunk,skip", [(1, 1, True), (3, 1, True), (2, 5, False), (8, 4, True)])
def test_stitching_covers_each_frame_once(first, chunk, skip):
    """Fake decoder: frame k decodes to 2048 samples of value k. Stitched output must be k = start..n-1."""
    cfg = C.ChunkConfig(first_chunk_frames=first, chunk_frames=chunk, skip_first_frame=skip)
    p, n_total, out, first_win = ChunkPlanner(cfg), 37, [], None
    for n in range(1, n_total + 1):          # frames arrive one at a time
        for w in p.plan(n):
            first_win = first_win or w
            assert w.decode_end <= n
            out.append(np.repeat(np.arange(w.decode_start, w.decode_end), C.SAMPLES_PER_FRAME)[w.keep_samples])
    for w in p.plan(n_total, final=True):
        out.append(np.repeat(np.arange(w.decode_start, w.decode_end), C.SAMPLES_PER_FRAME)[w.keep_samples])
    start = 1 if skip else 0
    assert np.array_equal(np.concatenate(out), np.repeat(np.arange(start, n_total), C.SAMPLES_PER_FRAME))
    # first chunk carries `first` frames and needs right_context frames after it
    assert first_win.emit_end - first_win.emit_start == first
    assert first_win.decode_end == start + first + cfg.right_context
