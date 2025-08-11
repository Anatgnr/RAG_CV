import requests
import json
from get_API_key import get_api_key

API_KEY = get_api_key()

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_text(prompt: str) -> str:
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are an assistant that replies only in valid JSON format, without any explanations."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def extract_cv_info(text: str) -> dict:
    prompt = f"""
    You are a HR expert analysing a CV. ONLY return a valid JSON object with the following structure:
    {{
    "skills": [string],
    "experiences": [string],
    "formations": [string]
    }}
    Make sure the value of each key is a JSON array (list) of strings.
    Do not include any extra text or comments.
    Extract key-words for each section from the following text below and **return only the JSON**:
    \"\"\"{text}\"\"\"
    """
    output = generate_text(prompt)
    try:
        cleaned = output.strip()
        # print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        # print(f"Output final for the CV is:\n{cleaned}")
        # print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output for CV was:\n{output}")
        return {}

def extract_job_info(text: str) -> dict:
    prompt = f"""
    You are a HR expert analysing a job offer. ONLY return a valid JSON object with the following structure:
    {{
    "skills": [string],
    "experiences": [string],
    "formations": [string]
    }}
    Make sure the value of each key is a JSON array (list) of strings.
    Do not include any extra text or comments.
    Extract key-words for each section from the following text below and **return only the JSON**:
    \"\"\"{text}\"\"\"
    """
    output = generate_text(prompt)
    try:
        cleaned = output.strip()
        # print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        # print(f"Output final for the job offer is:\n{cleaned}")
        # print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Output for job offer was:\n{output}")
        return {}

def reformulate_cv_for_job(cv: str, job_offer: str, job_structured: dict) -> str:
    """
    Reformule le CV en fonction de l'offre d'emploi en respectant les contraintes :
    - Ne pas inventer de compétences.
    - Peut déduire des soft skills raisonnablement.
    - Adapter le vocabulaire des expériences au contexte du job.
    Retourne un texte de CV reformulé prêt à être converti en PDF.
    """
    prompt = f"""
    Here are a CV: {cv}
    And here is the job offer: {job_offer}
    You are an expert HR assistant. Given the following CV data and job offer requirements:

    Job_Offer_importants_keywords:
    {job_structured}

    Task:
    Rewrite and adapt the CV to better match the job offer, by:
    - Reformulating experiences to highlight relevant skills and keywords.
    - Deduce soft skills reasonably (e.g., team projects imply teamwork).
    - Do NOT invent or add skills not present in the CV.
    - Only use skills and experiences explicitly or implicitly present in the CV.
    - Use language that aligns with the job offer terminology.
    - Output the reformulated CV as plain text, structured by sections: Skills, Experiences, Formations.
    - Do NOT add any explanations or commentary, ONLY the reformulated CV text.

    Please produce the reformulated CV now. It has to be relevant to the job offer and it shouldn't be a structure. 
    It should read like a natural text with sections and paragraphs. Use the structure of the original CV as a guide.
    It shouldn't not look like a list. Return only the reformulated CV as markdown text.
    """
    reformulated_cv_text = generate_text(prompt)
    return reformulated_cv_text