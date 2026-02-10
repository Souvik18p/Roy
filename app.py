import os
import requests
import gradio as gr
from fastapi import FastAPI
import uvicorn

# ---------------- CONFIG ----------------

MODEL_ID = "souvik18/Roy-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = f"https://router.huggingface.co/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
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

    r = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text}"

    data = r.json()
    if isinstance(data, dict) and "error" in data:
        return f"HF Error: {data['error']}"

    return data[0].get("generated_text", str(data))

# ---------------- GRADIO ----------------

gradio_ui = gr.ChatInterface(fn=chat)

# ---------------- FASTAPI ----------------

app = FastAPI()

app = gr.mount_gradio_app(app, gradio_ui, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
