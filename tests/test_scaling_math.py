"""CPU-only checks for the core formulas in scripts/scaling_math.py (no GPU, standard library)."""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scaling_math", Path(__file__).resolve().parent.parent / "scripts" / "scaling_math.py")
sm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sm)

N = 4.8e9
SHAPE = sm.MODEL_SHAPES["4.8B"]


def test_state_is_16_bytes_per_param():
    st = sm.train_state_bytes(1_000_000)
    assert st["total"] == 16 * 1_000_000
    assert st["optimizer"] == 12 * 1_000_000


def test_orpheus_shape_is_about_3b():
    sh = sm.MODEL_SHAPES["Orpheus 3B (reference)"]
    n = sm.params_from_shape(sh["L"], sh["d"], sh["ffn"], sh["heads"], sh["kv"])
    assert 3.2e9 < n < 3.4e9


def test_six_n_d_and_gpu_hours():
    # 1e9 params, 1e12 tokens -> 6e21 FLOPs
    assert sm.train_flops(1e9, 1e12) == pytest.approx(6e21)
    eff = sm.PEAK_BF16_TFLOPS * 1e12 * sm.MFU
    assert sm.gpu_hours(eff * 3600) == pytest.approx(1.0)  # one hour of effective FLOPs = 1 GPU-hour


def test_checkpointing_saves_activation_memory():
    full = sm.activation_bytes(SHAPE, 4096, 2, False)
    ckpt = sm.activation_bytes(SHAPE, 4096, 2, True)
    assert ckpt < full
    # layer part only: L*34 vs L*2 + 34 (units of s*b*d)
    logits = sm.logits_bytes(4096 * 2)
    sbd = 4096 * 2 * SHAPE["d"]
    assert full - logits == SHAPE["L"] * sm.ACT_COEF * sbd
    assert ckpt - logits == (SHAPE["L"] * sm.CKPT_SAVED_BYTES + sm.ACT_COEF) * sbd


def test_fsdp_shards_state_and_ddp_does_not():
    ddp = sm.per_gpu_memory(N, SHAPE, dict(kind="ddp", gpus=16, tp=1))
    f8 = sm.per_gpu_memory(N, SHAPE, dict(kind="fsdp", gpus=8, tp=1))
    f16 = sm.per_gpu_memory(N, SHAPE, dict(kind="fsdp", gpus=16, tp=1))
    assert ddp["state"] == pytest.approx(16 * N)
    assert f8["state"] == pytest.approx(16 * N / 8)
    assert f16["state"] == pytest.approx(16 * N / 16)
    hsdp = sm.per_gpu_memory(N, SHAPE, dict(kind="hsdp", gpus=16, tp=1))
    assert hsdp["state"] == pytest.approx(f8["state"])  # HSDP shards within a node only


def test_tp_divides_state_and_activations():
    f = sm.per_gpu_memory(N, SHAPE, dict(kind="fsdp", gpus=16, tp=1))
    t = sm.per_gpu_memory(N, SHAPE, dict(kind="fsdp", gpus=16, tp=4))
    assert t["state"] == pytest.approx(f["state"])        # 16N / (tp * gpus/tp) is the same total shard
    assert t["act_ckpt"] == pytest.approx(f["act_ckpt"] / 4)


def test_fsdp_comm_is_1_5x_ddp_in_the_large_group_limit():
    d = dict(kind="ddp", gpus=1024, tp=1)
    f = dict(kind="fsdp", gpus=1024, tp=1)
    dv = sum(sm.comm_volumes(N, SHAPE, d))
    fv = sum(sm.comm_volumes(N, SHAPE, f))
    assert fv / dv == pytest.approx(1.5, rel=0.01)


def test_hsdp_sends_less_across_nodes_than_flat_fsdp():
    intra_f, inter_f = sm.comm_volumes(N, SHAPE, dict(kind="fsdp", gpus=16, tp=1))
    intra_h, inter_h = sm.comm_volumes(N, SHAPE, dict(kind="hsdp", gpus=16, tp=1))
    assert inter_h < inter_f / 10
    assert intra_h > 0 and intra_f == 0
    # exact: 2 * (R-1)/R * (2N/8) with R=2
    assert inter_h == pytest.approx(2 * 0.5 * (2 * N / 8))
    # one node: no cross-node traffic at all
    assert sm.comm_volumes(N, SHAPE, dict(kind="fsdp", gpus=8, tp=1))[1] == 0


def test_gpus_needed_rounds_up():
    assert sm.gpus_needed(100, 6) == 17
    assert sm.gpus_needed(1000, 6) == 167
    assert sm.gpus_needed(12, 6) == 2


def test_interp_limit():
    pts = [(1, 8.0), (16, 10.0), (32, 14.0)]
    assert sm.interp_limit(pts, 12.0) == pytest.approx(24.0)
    assert sm.interp_limit(pts, 100.0) == 32


def test_measured_inputs_load_and_are_consistent():
    m = sm.load_measured()
    assert m["max_streams"] in m["passing_levels"]
    assert m["pipeline_cpu_headroom_x"] > 1
    r = sm.compute(m)
    assert r["data"]["tokens"] == pytest.approx(sm.DATASET_HOURS * m["tokens_per_audio_hour"])
    assert r["pipeline"]["l4_hours"] == pytest.approx(sm.DATASET_HOURS / (m["pipeline_hours_per_min"] * 60))


def test_doc_marker_replacement_is_idempotent(tmp_path):
    doc = tmp_path / "S.md"
    sm.update_doc("first", doc)
    sm.update_doc("second", doc)
    text = doc.read_text()
    assert text.count(sm.DOC_BEGIN) == 1 and "second" in text and "first" not in text
