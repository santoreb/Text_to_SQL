#!/usr/bin/env python3
"""
Generate SQL predictions from a fine-tuned CodeLlama model.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from tqdm import tqdm

# ---------------------------
# Configs
# ---------------------------
BASE_MODEL = "meta-llama/CodeLlama-7b-hf"
ADAPTER_PATH = "/home/jjvyas1/codellama_finetune/outputs/codellama-spider-lora"
DEV_FILE = "/home/jjvyas1/codellama_finetune/processed_spider/dev.jsonl"
OUT_FILE = "/home/jjvyas1/codellama_finetune/outputs/predictions.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load model + tokenizer
# ---------------------------
print("🔹 Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ---------------------------
# Generate SQL for each dev sample
# ---------------------------
gen_config = GenerationConfig(
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

records = [json.loads(l) for l in open(DEV_FILE)]
preds = []

print(f"🧠 Generating SQL for {len(records)} samples...")
for rec in tqdm(records):
    inp = rec["input"].strip()

    # Model prompt (consistent with your training)
    prompt = f"{inp}\n\nRETURN SQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract SQL after "RETURN SQL:"
    if "RETURN SQL:" in text:
        text = text.split("RETURN SQL:")[-1].strip()

    preds.append({
        "question": inp,
        "predicted_sql": text,
        "db_id": rec.get("db_id", "")
    })

# ---------------------------
# Save predictions
# ---------------------------
with open(OUT_FILE, "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")

print(f"\n✅ Saved predictions to: {OUT_FILE}")

