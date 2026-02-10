import os
import requests
import gradio as gr

MODEL_ID = "souvik18/Roy-v1"

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ CORRECT ENDPOINT FOR NORMAL LLM
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

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
            "top_p": 0.9,
            "do_sample": True
        }
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"Request failed: {str(e)}"

    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text}"

    data = r.json()

    # Different models return different format
    try:
        if isinstance(data, list):
            return data[0]["generated_text"]
        else:
            return str(data)
    except:
        return str(data)


demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Text Generation Model"
)

port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
