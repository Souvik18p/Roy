import os
import gradio as gr
import requests

# ---------------- CONFIG ----------------

MODEL_ID = "souvik18/Roy-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = f"https://router.huggingface.co/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- CHAT FUNCTION ----------------

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

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"Request failed: {str(e)}"

    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Exception:
        return f"Non-JSON response: {r.text}"

    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        if "loading" in str(err).lower():
            return "⏳ Roy is waking up... try again in 10 seconds."
        return f"HF Error: {err}"

    try:
        return data[0]["generated_text"]
    except Exception:
        return str(data)

# ---------------- GRADIO UI ----------------

demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Running via HuggingFace API"
)

# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
