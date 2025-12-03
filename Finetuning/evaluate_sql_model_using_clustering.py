#!/usr/bin/env python3
"""
Cluster-aware evaluation:
For EACH dev sample:
  1. Extract question
  2. Route question to nearest cluster (using centroids)
  3. Load cluster-specific FIRST-EPOCH LoRA
  4. Generate SQL
  5. Compute EM and EX

Outputs:
  - Overall Exact Match (EM)
  - Overall Execution Accuracy (EX)
"""

import os, json, sqlite3, sqlparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG PATHS
# -----------------------------
BASE_MODEL = "meta-llama/CodeLlama-7b-hf"

# centroids created earlier
CLUSTER_DIR = "/home/jjvyas1/codellama_finetune/processed_spider/clusters"
CENTROID_PATH = f"{CLUSTER_DIR}/cluster_centroids.npy"

# cluster-specific trained LoRA models
LORA_BASE = "/home/jjvyas1/codellama_finetune/outputs"

# full dev gold data
DEV_FILE = "/home/jjvyas1/codellama_finetune/processed_spider/dev.jsonl"

# spider DB folder
DB_DIR = "/home/jjvyas1/codellama_finetune/database"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Helper functions
# -----------------------------
def extract_question(input_text):
    try:
        return input_text.split("QUESTION:")[1].split("RETURN SQL:")[0].strip()
    except:
        return input_text.strip()


def normalize_sql(sql):
    sql = sql.strip().rstrip(";")
    sql = sqlparse.format(sql, keyword_case="lower", reindent=True)
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


# -----------------------------
# Load centroids and embedder
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
centroids = np.load(CENTROID_PATH)

def route_cluster(question):
    vec = embedder.encode([question], normalize_embeddings=True)
    dists = np.linalg.norm(centroids - vec, axis=1)
    return int(np.argmin(dists))


# -----------------------------
# Load schema for DB_ID
# -----------------------------
def get_schema_text(db_id):
    for line in open(DEV_FILE):
        obj = json.loads(line)
        if obj["db_id"] == db_id:
            return obj["input"].split("QUESTION:")[0].strip()
    return None


# -----------------------------
# Load cluster-specific LoRA
# -----------------------------
cluster_model_cache = {}

def load_model(cluster_id):
    """Load the earliest (lowest-numbered) checkpoint for this cluster."""
    if cluster_id in cluster_model_cache:
        return cluster_model_cache[cluster_id]

    cluster_path = f"{LORA_BASE}/lora_cluster_{cluster_id}"

    # List all subfolders that look like checkpoint folders
    ckpt_folders = [
        d for d in os.listdir(cluster_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(cluster_path, d))
    ]
    if not ckpt_folders:
        raise ValueError(f"No checkpoints found in {cluster_path}")

    # Sort numerically: checkpoint-2156 < checkpoint-2695 < checkpoint-3234
    ckpt_folders = sorted(ckpt_folders, key=lambda x: int(x.split("-")[1]))

    # ALWAYS pick the smallest-number checkpoint
    chosen_checkpoint = ckpt_folders[0]
    lora_dir = os.path.join(cluster_path, chosen_checkpoint)

    print(f"🔹 Using cluster {cluster_id} checkpoint: {chosen_checkpoint}")

    # Load tokenizer from cluster root (NOT checkpoint folder)
    tokenizer = AutoTokenizer.from_pretrained(cluster_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    cluster_model_cache[cluster_id] = (model, tokenizer)
    return model, tokenizer


# -----------------------------
# Prediction
# -----------------------------
def generate_sql(question, db_id):
    cid = route_cluster(question)
    model, tokenizer = load_model(cid)

    schema_text = get_schema_text(db_id)
    if schema_text is None:
        return "", cid

    prompt = f"{schema_text}\n\nQUESTION: {question}\n\nRETURN SQL:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    gen_cfg = GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "RETURN SQL:" in text:
        sql = text.split("RETURN SQL:")[-1].strip()
    else:
        sql = text.strip()

    return sql, cid


# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------
def evaluate():
    gold_data = [json.loads(l) for l in open(DEV_FILE)]
    total = len(gold_data)

    em_correct = 0
    ex_correct = 0

    print(f"🔍 Evaluating {total} dev samples with cluster routing...\n")

    for g in tqdm(gold_data, total=total):
        gold_sql = g["target"].strip()
        db_id = g["db_id"].strip()
        question = extract_question(g["input"])

        pred_sql, cid = generate_sql(question, db_id)

        # Exact match
        if exact_match(pred_sql, gold_sql):
            em_correct += 1

        # Execution accuracy
        db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(DB_DIR, db_id, f"{db_id}.db")

        if os.path.exists(db_path):
            if execution_match(pred_sql, gold_sql, db_path):
                ex_correct += 1

    em = em_correct / total * 100
    ex = ex_correct / total * 100

    print("\n================== RESULTS ==================")
    print(f"Exact Match (EM):        {em:.2f}%")
    print(f"Execution Accuracy (EX): {ex:.2f}%")
    print("================================================\n")


if __name__ == "__main__":
    evaluate()

