# ast_cluster_spider.py
# Purpose: AST-token embeddings -> clustering -> representative selection -> cleaned Spider train JSON
# Requirements:
# pip install sqlglot sentence-transformers hdbscan scikit-learn pandas pyarrow numpy tqdm

import json
import os
from sqlglot import parse_one, exp
from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import argparse
import pandas as pd
import trc

# ---------- Config / CLI ----------
parser = argparse.ArgumentParser(description="AST-based clustering for Spider SQL")
parser.add_argument("--spider_train", type=str, required=True, help="Path to spider train json (e.g., train_spider.json)")
parser.add_argument("--out_dir", type=str, default="ast_cluster_out", help="Output directory")
parser.add_argument("--embed_model", type=str, default="thenlper/gte-large", help="SBERT model for embeddings")
parser.add_argument("--use_trc", action="store_true", help="Include token role-context (TRC) in AST tokens (adds parent-arg role to node tokens)")
args = parser.parse_args()

# ---------- Load Spider train JSON (list of examples) ----------
print("Loading Spider training file:", args.spider_train)
spider_train = []
with open(args.spider_train, "r", encoding="utf8") as f:
    if args.spider_train.endswith(".jsonl"):
        for line in f:
            if line.strip():
                spider_train.append(json.loads(line))
    else:
        spider_train = json.load(f)  # expects list of dicts per Spider format


print(f"Loaded {len(spider_train)} examples from Spider training set.")
sqls = []
indices = []  # map back to examples
failed_trc = 0

print("Generating TRC strings for each example...")
for idx, ex in enumerate(tqdm(spider_train, desc="TRC", unit="ex")):
    sql = ex.get("target") or ex.get("query") or ex.get("sql")
    if not sql:
        failed_trc += 1
        continue
    try:
        trc_tokens = trc.sql_to_trc(sql)
        if trc_tokens == "UNHANDLED_SQL":
            failed_trc += 1
        ex["ast_trc"] = trc_tokens
    except Exception:
        trc_tokens = "TRC_ERROR"
        failed_trc += 1
    sqls.append(sql)
    indices.append(idx)

print(f"Found {len(sqls)} SQL queries in the provided file.")
if failed_trc:
    print(f"Warning: failed to generate TRC for {failed_trc} examples (see 'ast_trc' field for diagnostics).")

os.makedirs(args.out_dir, exist_ok=True)
input_name = os.path.basename(args.spider_train)
stem, ext = os.path.splitext(input_name)
output_path = os.path.join(args.out_dir, f"{stem}_with_trc{ext}")

print("Writing augmented dataset to:", output_path)
if args.spider_train.endswith(".jsonl"):
    with open(output_path, "w", encoding="utf8") as out_f:
        for ex in spider_train:
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
else:
    with open(output_path, "w", encoding="utf8") as out_f:
        json.dump(spider_train, out_f, ensure_ascii=False, indent=2)

if spider_train:
    print("Example with TRC:")
    print(spider_train[0])

