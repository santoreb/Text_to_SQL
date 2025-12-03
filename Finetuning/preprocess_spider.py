#!/usr/bin/env python3
"""
Preprocess Spider dataset into JSONL for CodeLlama / text-to-SQL fine-tuning.

Outputs:
  <out_dir>/train.jsonl
  <out_dir>/dev.jsonl

Each JSONL line:
  {
    "input": "DB_SCHEMA...QUESTION...RETURN SQL:",
    "target": "SELECT ...",
    "db_id": "..."
  }

Enhancements:
- Canonical whitespace and lowercase SQL
- Fully qualified foreign keys (table.col = table.col)
- Optional value delexicalization (--do_delex)
- Safer column and schema serialization
"""
import argparse
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm


# -------------------------------
# Helpers
# -------------------------------
def load_tables(tables_json_path):
    """Load and index tables.json by db_id"""
    with open(tables_json_path) as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def serialize_schema(entry):
    """
    Create human-readable schema text:
    tables, columns, foreign keys, column types, primary keys
    """
    db_id = entry["db_id"]
    tables = entry.get("table_names_original", [])
    cols = entry.get("column_names_original", [])
    types = entry.get("column_types", [])
    pk = entry.get("primary_keys", [])
    fk = entry.get("foreign_keys", [])

    # collect columns per table
    cols_per_table = defaultdict(list)
    for idx, (tid, cname) in enumerate(cols):
        if tid == -1:
            continue
        cols_per_table[tid].append((idx, cname))

    lines = [f"DB_ID: {db_id}", "SCHEMA:"]
    for i, tname in enumerate(tables):
        colnames = [c for _, c in cols_per_table[i]]
        coltext = ", ".join(colnames)
        lines.append(f"  {tname}({coltext})")

    # foreign keys: qualify table.col = table.col
    if fk:
        lines.append("FOREIGN_KEYS:")
        for (child_idx, parent_idx) in fk:
            child_tid, child_col = cols[child_idx]
            parent_tid, parent_col = cols[parent_idx]
            if child_tid == -1 or parent_tid == -1:
                continue
            left = f"{tables[child_tid]}.{child_col}"
            right = f"{tables[parent_tid]}.{parent_col}"
            lines.append(f"  {left} = {right}")

    # column types
    if types:
        lines.append("COLUMN_TYPES:")
        for (tid, cname), ctype in zip(cols, types):
            if tid == -1:
                continue
            lines.append(f"  {cname}: {ctype}")

    # primary keys (list of indices)
    if pk:
        pk_names = []
        for idx in pk:
            tid, cname = cols[idx]
            if tid == -1:
                continue
            pk_names.append(f"{tables[tid]}.{cname}")
        lines.append("PRIMARY_KEYS: " + ", ".join(pk_names))

    return "\n".join(lines)


# simple regex for numbers/strings/dates
_value_pattern = re.compile(r'\"[^\"]+\"|\'.+?\'|\b\d{4}-\d{2}-\d{2}\b|\b\d+\b')


def delexicalize_question(question: str):
    """
    Replace literal values (numbers, quoted strings) with placeholders [VAL_i]
    Returns (delexed_question, mapping_dict)
    """
    mapping = {}

    def repl(m):
        val = m.group(0)
        key = f"[VAL_{len(mapping) + 1}]"
        mapping[key] = val
        return key

    new_q = _value_pattern.sub(repl, question)
    return new_q, mapping


def normalize_sql(sql: str) -> str:
    """
    Canonicalize SQL: lowercase keywords, collapse spaces.
    """
    sql = sql.strip()
    # lowercase keywords safely
    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "ORDER BY", "GROUP BY",
        "HAVING", "LIMIT", "AND", "OR", "ASC", "DESC", "COUNT", "AVG",
        "SUM", "MIN", "MAX"
    ]
    for kw in sorted(keywords, key=len, reverse=True):
        sql = re.sub(rf"\b{kw}\b", kw.lower(), sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


def process_split(json_path, tables_map, out_path, do_delex=False):
    with open(json_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as outf:
        for ex in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
            db_id = ex["db_id"]
            question = ex.get("question", "").strip()
            sql = ex.get("query", "").strip()
            if not question or not sql:
                continue

            entry = tables_map.get(db_id)
            if not entry:
                print(f"Warning: missing schema for {db_id}")
                continue

            schema_text = serialize_schema(entry)

            # Delexicalize if requested
            values_block = ""
            if do_delex:
                dq, mapping = delexicalize_question(question)
                if mapping:
                    values_block = "\nVALUES:\n" + "\n".join(
                        [f"{k} = {v}" for k, v in mapping.items()]
                    )
                question = dq

            sql_norm = normalize_sql(sql)
            input_text = f"{schema_text}\n\nQUESTION: {question}{values_block}\n\nRETURN SQL:"
            rec = {
                "input": input_text,
                "target": sql_norm,
                "db_id": db_id
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------------
# Entry point
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", required=True, help="Path to spider directory")
    parser.add_argument("--out_dir", default="./processed_spider", help="Output folder")
    parser.add_argument("--do_delex", action="store_true", help="Apply delexicalization")
    args = parser.parse_args()

    tables_path = os.path.join(args.spider_dir, "tables.json")
    train_json = os.path.join(args.spider_dir, "train_spider.json")
    dev_json = os.path.join(args.spider_dir, "dev.json")
    if not os.path.exists(train_json):
        train_json = os.path.join(args.spider_dir, "train.json")

    assert os.path.exists(tables_path), "tables.json missing!"
    assert os.path.exists(train_json), "train_spider.json or train.json missing!"
    assert os.path.exists(dev_json), "dev.json missing!"

    tables_map = load_tables(tables_path)
    os.makedirs(args.out_dir, exist_ok=True)

    process_split(train_json, tables_map, os.path.join(args.out_dir, "train.jsonl"), do_delex=args.do_delex)
    process_split(dev_json, tables_map, os.path.join(args.out_dir, "dev.jsonl"), do_delex=args.do_delex)
    print(f"✅ Done! Files written to {args.out_dir}")


if __name__ == "__main__":
    main()
