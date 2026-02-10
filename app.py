import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_ID = "souvik18/Roy-v1"

print("Loading Roy-v1...")

# 4-bit quantization (works on free CPU)
bnb = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto"
)

# avoid pad warning
model.generation_config.pad_token_id = tokenizer.eos_token_id


def chat(message, history):
    prompt = f"[INST] {message} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = inputs.to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return reply


demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Personal AI Assistant"
)

# ✅ RENDER PORT FIX
port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
