"""
将多个 JSON 文件转为多个小 parquet 文件（嵌套 Arrow struct 格式）。
每个 parquet 文件只包含一个 row group，行数由 ROWS_PER_FILE 控制，
确保单个列 chunk 不超过 pyarrow 2GB 限制。

用法:
    python convert_json_to_parquet.py \
        --input_dir /llm-align/liuchonghan/ins_dataset/ins_dataset \
        --output_dir /llm-align/liuchonghan/ins_dataset/ins_dataset \
        --rows_per_file 50000
"""

import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def detect_keys(record):
    if "question" in record and "response" in record:
        return "question", "response"
    if "prompt" in record and "res" in record:
        return "prompt", "res"
    if "prompt" in record and "response" in record:
        return "prompt", "response"
    raise ValueError(f"无法识别的 key 组合: {list(record.keys())}")


def build_messages(record, q_key, a_key):
    return [
        {"role": "user", "content": str(record[q_key])},
        {"role": "assistant", "content": str(record[a_key])},
    ]


MESSAGES_TYPE = pa.list_(
    pa.struct([("role", pa.utf8()), ("content", pa.utf8())])
)


def flush_batch(batch, output_dir, file_idx):
    arr = pa.array(batch, type=MESSAGES_TYPE)
    table = pa.table({"messages": arr})
    path = os.path.join(output_dir, f"train_part{file_idx:04d}.parquet")
    pq.write_table(table, path, row_group_size=len(batch))
    print(f"  写入 {path}  ({len(batch)} 行)")
    return file_idx + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_file", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    batch = []
    file_idx = 0
    total_rows = 0

    for jf in json_files:
        print(f"处理: {os.path.basename(jf)}")
        with open(jf, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.split("\n") if line.strip()]

        if not records:
            continue

        q_key, a_key = detect_keys(records[0])

        for rec in records:
            msgs = build_messages(rec, q_key, a_key)
            batch.append(msgs)
            total_rows += 1

            if len(batch) >= args.rows_per_file:
                file_idx = flush_batch(batch, args.output_dir, file_idx)
                batch = []

    if batch:
        file_idx = flush_batch(batch, args.output_dir, file_idx)

    print(f"\n完成! 共 {total_rows} 行, 写入 {file_idx} 个 parquet 文件")


if __name__ == "__main__":
    main()
