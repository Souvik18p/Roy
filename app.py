import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_ID = "souvik18/Roy-v1"

# 4bit for low cost hosting
bnb = BitsAndBytesConfig(load_in_4bit=True)

print("Loading Roy-v1...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto"
)

model.generation_config.pad_token_id = tokenizer.eos_token_id

def chat(message, history):
    prompt = f"[INST] {message} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

demo.launch(server_name="0.0.0.0", server_port=7860)
