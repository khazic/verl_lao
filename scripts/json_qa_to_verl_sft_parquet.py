#!/usr/bin/env python3
"""
Convert a QA-style dataset (JSON array or JSONL) into a VERL SFT parquet file.

Input item example:
  {"question": "...", "response": "..."}

Output schemas:
- single_turn: columns `question` and `answer` (strings)
  Use with `verl/trainer/config/sft_trainer.yaml` defaults:
    data.prompt_key=question
    data.response_key=answer

- messages: column `messages` (list of {role, content})
  Use with `verl/trainer/config/sft_trainer_engine.yaml` (MultiTurnSFTDataset):
    data.messages_key=messages
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Optional


def iter_items(path: str) -> Iterator[Dict[str, Any]]:
    """
    Iterate items from either:
    - JSON array file: [ {...}, {...}, ... ]
    - JSONL file: one JSON object per line

    For huge JSON arrays, install `ijson` to stream:
      pip install ijson
    """
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None

    with open(path, "rb") as f:
        # Peek the first non-whitespace byte.
        first = None
        while True:
            b = f.read(1)
            if not b:
                break
            if b not in b" \t\r\n":
                first = b
                break
        f.seek(0)

        if first == b"[":
            if ijson is None:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected a JSON array in {path}")
                for obj in data:
                    if not isinstance(obj, dict):
                        raise ValueError(f"Expected dict items, got {type(obj)}")
                    yield obj
                return

            for obj in ijson.items(f, "item"):
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected dict items, got {type(obj)}")
                yield obj
            return

        # JSONL fallback
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict items, got {type(obj)}")
            yield obj


def make_row(
    item: Dict[str, Any],
    *,
    input_key: str,
    output_key: str,
    out_format: str,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    q = item.get(input_key)
    a = item.get(output_key)
    if q is None or a is None:
        raise KeyError(f"Missing keys: {input_key!r} / {output_key!r}. Got keys={sorted(item.keys())}")
    if not isinstance(q, str) or not isinstance(a, str):
        raise TypeError(f"Expected strings; got {type(q)} / {type(a)}")

    if out_format == "single_turn":
        return {"question": q, "answer": a}

    if out_format == "messages":
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
        return {"messages": messages}

    raise ValueError(f"Unknown out_format: {out_format}")


def write_parquet(
    *,
    input_path: str,
    output_path: str,
    input_key: str,
    output_key: str,
    out_format: str,
    system_prompt: Optional[str],
    batch_size: int,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if out_format == "single_turn":
        schema = pa.schema([("question", pa.string()), ("answer", pa.string())])
    else:
        msg_struct = pa.struct([("role", pa.string()), ("content", pa.string())])
        schema = pa.schema([("messages", pa.list_(msg_struct))])

    writer: Optional[pq.ParquetWriter] = None
    buf: List[Dict[str, Any]] = []
    total = 0

    def flush() -> None:
        nonlocal writer, buf, total
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema=schema, compression="zstd")
        writer.write_table(table)
        total += len(buf)
        buf = []

    try:
        for it in iter_items(input_path):
            buf.append(
                make_row(
                    it,
                    input_key=input_key,
                    output_key=output_key,
                    out_format=out_format,
                    system_prompt=system_prompt,
                )
            )
            if len(buf) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    return total


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSON/JSONL path")
    ap.add_argument("--output", required=True, help="Output parquet path")
    ap.add_argument("--input_key", default="question", help="Field name for prompt text")
    ap.add_argument("--output_key", default="response", help="Field name for response text")
    ap.add_argument("--format", dest="out_format", choices=["single_turn", "messages"], default="single_turn")
    ap.add_argument("--system_prompt", default=None, help="Optional system prompt (messages format only)")
    ap.add_argument("--batch_size", type=int, default=4096, help="Write batch size")
    args = ap.parse_args()

    n = write_parquet(
        input_path=args.input,
        output_path=args.output,
        input_key=args.input_key,
        output_key=args.output_key,
        out_format=args.out_format,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
    )
    print(f"[OK] Wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()

