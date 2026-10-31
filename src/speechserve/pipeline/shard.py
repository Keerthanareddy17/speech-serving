"""Shard stage: SNAC codes -> Orpheus token ids (7 per frame, same layout and offsets the server decodes),
written with the text and durations into fixed-size parquet shards, plus a manifest.

Shard columns: id, speaker_id, chapter_id, text, duration_s (trimmed), orig_duration_s, n_frames,
n_tokens, tokens (list<int32>, ids in [128266, 128266 + 7*4096)).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..codec import frames_to_token_ids, layers_to_frames

SR = 24_000


class ShardWriter:
    def __init__(self, out_dir: Path, clips_per_shard: int = 2500):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.clips_per_shard = clips_per_shard
        self.rows: list[dict] = []
        self.shards: list[dict] = []
        self.write_s = 0.0
        self.clips = self.tokens = 0
        self.audio_s = 0.0
        self.mismatches: list[str] = []

    def add(self, clip, l0, l1, l2) -> None:
        t = time.perf_counter()
        ids = frames_to_token_ids(layers_to_frames(l0, l1, l2)).astype(np.int32)
        n_frames = len(l0)
        dur = len(clip.audio) / SR
        # token count must match duration: 7 tokens per started 2048-sample frame
        if len(ids) != 7 * -(-len(clip.audio) // 2048):
            self.mismatches.append(clip.id)
        self.rows.append({"id": clip.id, "speaker_id": clip.speaker_id, "chapter_id": clip.chapter_id,
                          "text": clip.text, "duration_s": round(dur, 4), "orig_duration_s": round(clip.orig_s, 4),
                          "n_frames": n_frames, "n_tokens": len(ids), "tokens": ids})
        self.clips += 1
        self.tokens += len(ids)
        self.audio_s += dur
        self.write_s += time.perf_counter() - t
        if len(self.rows) >= self.clips_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        t = time.perf_counter()
        name = f"shard-{len(self.shards):05d}.parquet"
        cols = {k: [r[k] for r in self.rows] for k in self.rows[0] if k != "tokens"}
        cols["tokens"] = pa.array([r["tokens"] for r in self.rows], type=pa.list_(pa.int32()))
        table = pa.table(cols)
        pq.write_table(table, self.out_dir / name, compression="zstd")
        self.shards.append({"file": name, "clips": len(self.rows), "tokens": int(sum(r["n_tokens"] for r in self.rows)),
                            "audio_hours": round(sum(r["duration_s"] for r in self.rows) / 3600, 4),
                            "bytes": (self.out_dir / name).stat().st_size})
        self.rows = []
        self.write_s += time.perf_counter() - t

    def manifest(self, extra: dict) -> dict:
        self.flush()
        hours = self.audio_s / 3600
        m = {"clips": self.clips, "audio_hours": round(hours, 4), "tokens": self.tokens,
             "tokens_per_audio_hour": round(self.tokens / hours) if hours else None,
             "tokens_per_audio_second": round(self.tokens / self.audio_s, 3) if self.audio_s else None,
             "token_layout": "Orpheus: 7 ids per 2048-sample SNAC frame, id = code + 128266 + (pos % 7) * 4096",
             "token_count_mismatches": len(self.mismatches), "shards": self.shards, **extra}
        (self.out_dir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        return m
