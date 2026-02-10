import os
import gradio as gr
import requests

MODEL_ID = "souvik18/Roy-v1"

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def chat(message, history):

    prompt = f"[INST] {message} [/INST]"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    r = requests.post(API_URL, headers=headers, json=payload)

    try:
        return r.json()[0]["generated_text"]
    except:
        return str(r.json())


demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Running via HuggingFace API"
)

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
