#!/usr/bin/env python3
"""
Inference script for your fine-tuned CodeLlama LoRA model.
Type natural language questions — get SQL queries back.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# ---------------------------
# Paths and configs
# ---------------------------
BASE_MODEL = "meta-llama/CodeLlama-7b-hf"
ADAPTER_PATH = "/home/jjvyas1/codellama_finetune/outputs/codellama-spider-lora"   # path to your fine-tuned adapter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load model + tokenizer
# ---------------------------
print("🔹 Loading base model and LoRA adapter...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# Merge with LoRA weights
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model and tokenizer loaded successfully.\n")

# ---------------------------
# Interactive inference loop
# ---------------------------
print("💬 Enter natural language questions (type 'exit' to quit)\n")

while True:
    user_input = input("🧠 Question: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # Prepare prompt — same structure used during training
    prompt = f"Translate the following question into SQL:\n\nQuestion: {user_input}\nSQL:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generation configuration
    gen_config = GenerationConfig(
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=gen_config)

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract only SQL part (optional cleanup)
    if "SQL:" in output_text:
        output_text = output_text.split("SQL:")[-1].strip()

    print(f"🧾 Predicted SQL:\n{output_text}\n")

