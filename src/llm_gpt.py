from llama_cpp import Llama
import os
import json
import sys
import time
import threading
import torch


print(f"CUDA version: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

# MODEL_PATH = os.path.abspath(os.path.join("models"))
MODEL_PATH = "models/gpt-oss-20b-Q4_K_S.gguf"
print(f"Loading model from {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    ngpu_layers=-1,  # Use GPU if available
    n_ctx=2048,
    verbose=True
)

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

def extract_cv_info(text: str) -> dict:
    prompt = f"""
You are a HR you need to fill the following tabs with the infos of the cv you'll be given.
Pay attention to the details. Do a deep analysis understand the implicit implied.

{{
  "skills": [],
  "experiences": [],
  "formations": []
}}

Here is the CV :
{text}
"""

    loading = loading_animation()
    try:
        with llm.chat_session():
            output_chunks = []
            for chunk in llm.generate(prompt, max_tokens=1024):
                output_chunks.append(chunk)
            output = "".join(output_chunks)
    finally:
        loading.set()  # Stop loading animation

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
You are a HR you need to fill the following tabs with the infos of the job description you'll be given.
Pay attention to the details. Do a deep analysis understand the implicit implied.

{{
  "skills": [],
  "experiences": [],
  "formations": []
}}

Here is the job description :
{text}
"""

    loading = loading_animation()
    try:
        with llm.chat_session():
            output_chunks = []
            for chunk in llm.generate(prompt, max_tokens=1024):
                output_chunks.append(chunk)
            output = "".join(output_chunks)
    finally:
        loading.set()  # Stop loading animation

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output was:\n{output}")
        return {}
