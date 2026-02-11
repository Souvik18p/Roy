import os
import requests
import gradio as gr

MODEL_ID = "souvik18/Roy-v1"

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ CORRECT ROUTER ENDPOINT FOR TEXT MODELS
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def chat(message, history):

    prompt = f"[INST] {message} [/INST]"

    payload = {
        "inputs": prompt,

        "options": {
            "wait_for_model": True,
            "use_cache": False
        },

        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    except Exception as e:
        return f"Request failed: {str(e)}"

    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text}"

    data = r.json()

    try:
        if isinstance(data, list):
            return data[0].get("generated_text", str(data[0]))
        return str(data)
    except Exception as e:
        return f"Parse error: {str(e)} | Raw: {data}"


demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Running via HuggingFace Router Inference"
)

port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
