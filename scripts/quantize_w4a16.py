"""Step 1c: GPTQ W4A16 (group 128) quantization of Orpheus 3B with llm-compressor, calibrated on Orpheus outputs.

Runs in the separate quantization venv (llm-compressor's torch/transformers pins clash with vLLM's):
  /content/venvs/quant/bin/python scripts/quantize_w4a16.py             # lm_head kept bf16 (step 1c)
  /content/venvs/quant/bin/python scripts/quantize_w4a16.py --lm-head   # lm_head quantized too (step 1d)
Input : /content/drive/MyDrive/speech-infra/calib/orpheus_calib_128.jsonl  (from scripts/make_calib.py)
Output: /content/drive/MyDrive/speech-infra/models/orpheus-3b-w4a16/        (Drive, not git)
        results/step1c_quantize.json
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "canopylabs/orpheus-3b-0.1-ft"
TOKENIZER = "canopylabs/orpheus-3b-0.1-pretrained"
CALIB = Path("/content/drive/MyDrive/speech-infra/calib/orpheus_calib_128.jsonl")
OUT_DIR = Path("/content/drive/MyDrive/speech-infra/models/orpheus-3b-w4a16")
ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results/step1c_quantize.json"
OUT_DIR_LMHEAD = Path("/content/drive/MyDrive/speech-infra/models/orpheus-3b-w4a16-lmhead")
OUT_JSON_LMHEAD = ROOT / "results/step1d_quantize_lmhead.json"

# W4A16 preset: int4 symmetric weights, one scale per group of 128, activations stay 16-bit.
# Embeddings are not Linear layers, so they stay bf16; lm_head (tied to them) is skipped explicitly.
RECIPE = dict(targets="Linear", scheme="W4A16", ignore=["lm_head"])
MAX_SEQ_LEN = 2048  # longest calibration sequence is < 1k tokens


def add_lm_head_target(config_path):
    """Name lm_head in the quantization targets so vLLM finds it.

    llm-compressor matches "Linear" against HF's nn.Linear class, which includes lm_head. vLLM matches
    targets by layer name or by substring of its own class name, and its lm_head is a `ParallelLMHead`,
    which doesn't contain "Linear". Without this, vLLM loads lm_head as unquantized and fails on
    `lm_head.weight_packed`. Metadata only: the weights are unchanged.
    """
    cfg = json.loads(config_path.read_text())
    for group in cfg["quantization_config"]["config_groups"].values():
        if "lm_head" not in group["targets"]:
            group["targets"].append("lm_head")
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-head", action="store_true", help="also quantize lm_head (untied from embeddings)")
    args = parser.parse_args()
    recipe = {**RECIPE, "ignore": []} if args.lm_head else RECIPE
    out_dir, out_json = (OUT_DIR_LMHEAD, OUT_JSON_LMHEAD) if args.lm_head else (OUT_DIR, OUT_JSON)

    rows = [json.loads(line) for line in CALIB.read_text().splitlines()]
    ids = [r["prompt_ids"] + r["output_ids"] for r in rows]
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})

    # bf16 explicitly: the checkpoint is stored in fp32, and vLLM serves it in bf16.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if args.lm_head:
        # lm_head shares its matrix with the embedding table. Give it its own copy so GPTQ quantizes the
        # output projection while the embedding lookup stays bf16.
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
        model.config.tie_word_embeddings = False

    t0 = time.perf_counter()
    oneshot(model=model, dataset=ds, recipe=GPTQModifier(**recipe),
            max_seq_length=MAX_SEQ_LEN, num_calibration_samples=len(rows))
    quant_s = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    if args.lm_head:
        add_lm_head_target(out_dir / "config.json")
    qcfg = json.loads((out_dir / "config.json").read_text())["quantization_config"]
    weights_cfg = next(iter(qcfg["config_groups"].values()))["weights"]
    size_gib = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 2**30

    result = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL,
        "vllm_version": None,  # quantization runs in the llm-compressor venv, which has no vLLM
        "llmcompressor_version": version("llmcompressor"),
        "compressed_tensors_version": version("compressed-tensors"),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "python_version": platform.python_version(),
        "config": {"recipe": {"modifier": "GPTQModifier", **recipe}, "calibration": str(CALIB),
                   "num_calibration_samples": len(rows), "max_seq_length": MAX_SEQ_LEN, "dtype": "bfloat16"},
        "quantization_s": round(quant_s, 1),
        "output_dir": str(out_dir),
        "output_safetensors_gib": round(size_gib, 2),
        "saved_quant_format": qcfg.get("format"),
        "saved_weights_config": {k: weights_cfg.get(k) for k in ("num_bits", "type", "symmetric", "group_size", "strategy")},
        "saved_ignore": qcfg.get("ignore"),
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
