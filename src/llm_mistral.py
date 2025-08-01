from llama_cpp import Llama
import os
import json
import sys
import time
import threading

MODEL_PATH = os.path.abspath(os.path.join("models", "openhermes-2.5-Mistral-7b.Q4_K_S.gguf"))
print(f"Loading model from {MODEL_PATH}")

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=32)


def loading_animation(message="Thinking..."):
    stop_loading = threading.Event()
    def animate():
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while not stop_loading.is_set():
            print(f"\r{message} {spinner[idx % len(spinner)]}", end='', flush=True)
            idx += 1
            time.sleep(0.1)
        print('\r' + ' ' * (len(message) + 2) + '\r', end='', flush=True)
    t = threading.Thread(target=animate)
    t.start()
    return stop_loading

def generate_text(prompt, max_tokens=512):
    loading = loading_animation()
    try:
        response = llm(prompt, max_tokens=max_tokens)
        output = response['choices'][0]['text']
    finally:
        loading.set()  # Stop animation
    return output

def extract_cv_info(text: str) -> dict:
    prompt = f"""
You are a HR expert analysing a CV. ONLY return a valid JSON object with the following structure:
{{
  "skills": [],
  "experiences": [],
  "formations": []
}}
Extract the key-words for each section from the following text:
{text}
"""
    output = generate_text(prompt)
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output was:\n{output}")
        return {}

def extract_job_info(text: str) -> dict:
    prompt = f"""
You are a HR expert analysing a job offer. ONLY return a valid JSON object with the following structure:
{{
  "skills": [],
  "experiences": [],
  "formations": []
}}
Extract the key-words for each section from the following text:
{text}
"""
    output = generate_text(prompt)
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output was:\n{output}")
        return {}
