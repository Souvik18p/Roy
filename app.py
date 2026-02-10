import os
import requests
import gradio as gr

MODEL_ID = "souvik18/Roy-v1"

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ OFFICIAL TEXT GENERATION ENDPOINT
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def chat(message, history):

    prompt = f"[INST] {message} [/INST]"

    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True
        },
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
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
        # HF normal generation format
        if isinstance(data, list):
            return data[0]["generated_text"]
        return str(data)
    except:
        return str(data)


demo = gr.ChatInterface(
    fn=chat,
    title="Roy-v1 by Souvik",
    description="Running via HuggingFace Inference"
)

port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
