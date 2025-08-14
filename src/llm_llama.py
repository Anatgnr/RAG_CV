from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import sys
import time
import threading
import torch

os.environ["HF_HOME"] = "D:/code/hf_cache"
# Trop bete le modèle

print(f"CUDA version: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
model_id = "NousResearch/Nous-Hermes-13b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")

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
    message = f"""
    You are an HR assistant. Extract ONLY the structured information from the following CV.
    here is the CV: {text}

    Your task is to return a valid JSON object containing the following fields:
    - "skills": list of technical or soft skills explicitly mentioned
    - "experiences": job titles or roles mentioned
    - "formations": degrees, certifications, or educational info mentioned

    ⚠️ Do NOT infer or invent anything.
    ⚠️ Do NOT add any commentary or explanation.
    ⚠️ Output ONLY a valid JSON object — no markdown, no text, just JSON.
    ⚠️ Complete the JSON object fields with keywords only.

    Expected JSON object format:
    {{
    "skills": [ "skill1", "skill2", ... ],
    "experiences": [ "experience1", "experience2", ... ],
    "formations": [ "formation1", "formation2", ... ]
    }}

    Return ONLY the JSON object filled with the extracted information above:
    """

    loading = loading_animation()
    try:
        input_ids = tokenizer(message, return_tensors="pt")
        inputs_ids = {k: v.to(model.device) for k, v in input_ids.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            
    finally:
        loading.set()  # Stop loading animation

    try:
        with torch.no_grad():
            print(f"Raw Output : \n  {decoded}")
    finally:
        loading.set()  # Stop loading animation

    try:
        start = decoded.find("{")
        end = decoded.rfind("}") + 1
        return json.loads(decoded[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output was:\n{decoded}")
        return {}

def extract_job_info(text: str) -> dict:
    message = f"""
    You are an HR assistant. Extract ONLY the structured information from the following job offer.
    here is the job offer: {text}

    Your task is to return a valid JSON object containing the following fields:
    - "skills": list of technical or soft skills explicitly mentioned
    - "experiences": job titles or roles mentioned
    - "formations": degrees, certifications, or educational info mentioned

    ⚠️ Do NOT infer or invent anything.
    ⚠️ Do NOT add any commentary or explanation.
    ⚠️ Output ONLY a valid JSON object — no markdown, no text, just JSON.
    ⚠️ Complete the JSON object fields with keywords only.

    Expected format:
    {{
    "skills": [ "skill1", "skill2", ... ],
    "experiences": [ "experience1", "experience2", ... ],
    "formations": [ "formation1", "formation2", ... ]
    }}

    Return ONLY the JSON object filled with the extracted information above:
    """

    loading = loading_animation()
    try:
        input_ids = tokenizer(message, return_tensors="pt")
        inputs_ids = {k: v.to(model.device) for k, v in input_ids.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    finally:
        loading.set()  # Stop loading animation

    try:
        with torch.no_grad():
            print(f"Raw Output : \n  {decoded}")
    finally:
        loading.set()  # Stop loading animation

    try:
        start = decoded.find("{")
        end = decoded.rfind("}") + 1
        return json.loads(decoded[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output was:\n{decoded}")
        return {}
