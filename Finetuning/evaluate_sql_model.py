#!/usr/bin/env python3
"""
Evaluate text-to-SQL model on Exact Match (EM) and Execution Accuracy (EX)
"""

import json, os, sqlite3, sqlparse
from tqdm import tqdm


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = sqlparse.format(sql, reindent=True, keyword_case="lower")
    return " ".join(sql.split())


def exact_match(pred, gold):
    return normalize_sql(pred) == normalize_sql(gold)


def execute_sql(db_path, sql):
    if not os.path.exists(db_path):
        return False, []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, sorted(rows)
    except Exception:
        return False, []


def execution_match(pred_sql, gold_sql, db_path):
    ok_pred, pred_out = execute_sql(db_path, pred_sql)
    ok_gold, gold_out = execute_sql(db_path, gold_sql)
    if not ok_pred or not ok_gold:
        return False
    return pred_out == gold_out


def evaluate(pred_file, gold_file, db_dir):
    preds = [json.loads(l) for l in open(pred_file)]
    golds = [json.loads(l) for l in open(gold_file)]
    assert len(preds) == len(golds)

    total = len(golds)
    em_correct = 0
    ex_correct = 0

    for p, g in tqdm(zip(preds, golds), total=total, desc="Evaluating"):
        pred_sql = p["predicted_sql"].strip()
        gold_sql = g["target"].strip()
        db_id = g.get("db_id", "")
        #db_path = os.path.join(db_dir, f"{db_id}.db")
        # Build correct path for Spider-style folder layout
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            # Fallback for .db extension, just in case
            db_path = os.path.join(db_dir, db_id, f"{db_id}.db")

        if exact_match(pred_sql, gold_sql):
            em_correct += 1

        if db_id and os.path.exists(db_path):
            if execution_match(pred_sql, gold_sql, db_path):
                ex_correct += 1

    em = em_correct / total * 100
    ex = ex_correct / total * 100
    print(f"\n✅ Evaluation complete.")
    print(f"Exact Match (EM): {em:.2f}%")
    print(f"Execution Accuracy (EX): {ex:.2f}%\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--db_dir", required=True)
    args = parser.parse_args()
    evaluate(args.pred_file, args.gold_file, args.db_dir)

