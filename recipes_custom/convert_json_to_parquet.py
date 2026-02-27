"""
Convert JSON/JSONL files into sharded Parquet files with `messages` column.

Design goals:
- Stream records instead of loading full files into memory (OOM-safe for very large datasets).
- Write fixed-size parquet shards (`rows_per_file`) to cap memory and file size.
- Accept both JSONL and JSON array formats.

Example:
    python recipes_custom/convert_json_to_parquet.py \
        --input_dir /llm-align/liuchonghan/ins_dataset/ins_dataset \
        --output_dir /llm-align/liuchonghan/ins_dataset/ins_dataset \
        --rows_per_file 50000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


MESSAGES_TYPE = pa.list_(pa.struct([("role", pa.utf8()), ("content", pa.utf8())]))
KEY_CANDIDATES: Sequence[Tuple[str, str]] = (
    ("question", "response"),
    ("prompt", "res"),
    ("prompt", "response"),
)


def detect_keys(record: Dict) -> Tuple[str, str]:
    for q_key, a_key in KEY_CANDIDATES:
        if q_key in record and a_key in record:
            return q_key, a_key
    raise ValueError(f"Unsupported keys in record: {list(record.keys())}")


def build_messages(record: Dict, q_key: str, a_key: str):
    return [
        {"role": "user", "content": str(record[q_key])},
        {"role": "assistant", "content": str(record[a_key])},
    ]


def iter_json_array_records(path: str, chunk_size: int = 1024 * 1024) -> Iterator[Dict]:
    """Stream records from a top-level JSON array without loading whole file."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        # Skip leading whitespace and consume '['
        while True:
            ch = f.read(1)
            if ch == "":
                return
            if not ch.isspace():
                break
        if ch != "[":
            raise ValueError(f"Expected JSON array in {path}, got leading char: {ch!r}")

        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                buf += chunk

            made_progress = False
            while True:
                buf = buf.lstrip()
                if not buf:
                    break

                if buf[0] == ",":
                    buf = buf[1:]
                    made_progress = True
                    continue

                if buf[0] == "]":
                    return

                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break

                yield obj
                buf = buf[idx:]
                made_progress = True

            if not chunk:
                # EOF reached: allow only trailing whitespace and optional ']'
                if buf.strip() in ("", "]"):
                    return
                raise ValueError(f"Malformed JSON array near EOF: {path}")

            if not made_progress and len(buf) > 64 * 1024 * 1024:
                raise ValueError(
                    f"Single record appears too large (>64MB buffered) while parsing: {path}"
                )


def iter_jsonl_records(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON line at {path}:{line_no}: {e}") from e


def iter_records(path: str) -> Iterator[Dict]:
    # Detect format by first non-whitespace character.
    with open(path, "r", encoding="utf-8") as f:
        first = ""
        while True:
            ch = f.read(1)
            if ch == "":
                return
            if not ch.isspace():
                first = ch
                break

    if first == "[":
        yield from iter_json_array_records(path)
    else:
        yield from iter_jsonl_records(path)


def flush_batch(batch, output_dir: str, file_idx: int, compression: Optional[str]) -> int:
    arr = pa.array(batch, type=MESSAGES_TYPE)
    table = pa.table({"messages": arr})
    out_path = os.path.join(output_dir, f"train_part{file_idx:04d}.parquet")
    pq.write_table(
        table,
        out_path,
        row_group_size=len(batch),
        compression=compression,
        use_dictionary=False,
    )
    print(f"  wrote {out_path} ({len(batch)} rows)")
    return file_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_file", type=int, default=50000)
    parser.add_argument(
        "--input_glob",
        default="*.json",
        help="Glob pattern under input_dir. Example: '*.json' or '*.jsonl'",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--start_part_index",
        type=int,
        default=0,
        help="First output shard index. Useful for resume/manual split.",
    )
    args = parser.parse_args()

    if args.rows_per_file <= 0:
        raise ValueError("rows_per_file must be > 0")

    compression = None if args.compression == "none" else args.compression

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_glob)))
    print(f"found {len(json_files)} files in {args.input_dir} matching {args.input_glob!r}")
    if not json_files:
        return

    batch = []
    file_idx = args.start_part_index
    total_rows = 0
    skipped_rows = 0

    key_cache: Dict[Tuple[str, ...], Tuple[str, str]] = {}

    for jf in json_files:
        print(f"processing: {os.path.basename(jf)}")
        file_rows = 0
        for rec in iter_records(jf):
            if not isinstance(rec, dict):
                skipped_rows += 1
                continue

            record_keys = tuple(sorted(rec.keys()))
            qa_keys = key_cache.get(record_keys)
            if qa_keys is None:
                try:
                    qa_keys = detect_keys(rec)
                    key_cache[record_keys] = qa_keys
                except ValueError:
                    skipped_rows += 1
                    continue

            q_key, a_key = qa_keys
            try:
                batch.append(build_messages(rec, q_key, a_key))
            except Exception:
                skipped_rows += 1
                continue

            total_rows += 1
            file_rows += 1

            if len(batch) >= args.rows_per_file:
                file_idx = flush_batch(batch, args.output_dir, file_idx, compression)
                batch = []

        print(f"  accepted rows from file: {file_rows}")

    if batch:
        file_idx = flush_batch(batch, args.output_dir, file_idx, compression)

    print(
        f"done. total_rows={total_rows}, skipped_rows={skipped_rows}, parquet_files={file_idx - args.start_part_index}"
    )


if __name__ == "__main__":
    main()
