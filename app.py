import os
import requests
import gradio as gr
from fastapi import FastAPI
import uvicorn

# ---------------- CONFIG ----------------

MODEL_ID = "souvik18/Roy-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- CHAT FUNCTION ----------------

def chat(message, history):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": message}
        ]
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"Request failed: {str(e)}"

    if r.status_code != 200:
        # Include full error so you can debug if something goes wrong
        return f"HTTP {r.status_code}: {r.text}"

    data = r.json()

    # Extract the assistant (AI) response from the chat format
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)

# ---------------- GRADIO UI ----------------

gradio_ui = gr.ChatInterface(fn=chat, title="Roy-v1 by Souvik", description="Chat via HuggingFace Inference Providers")

# ---------------- FASTAPI ----------------

app = FastAPI()

# Mount Gradio under "/"
app = gr.mount_gradio_app(app, gradio_ui, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
