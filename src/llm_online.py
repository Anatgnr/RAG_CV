import requests
import json
from src.get_API_key import get_api_key

API_KEY = get_api_key()

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def clean_json_markdown(text: str) -> str:
    """
    Supprime les balises ```json et ``` autour du JSON si présentes.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def generate_key_words(prompt: str) -> str:
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "system", "content": "You are an HR expert assistant that replies \
                ONLY in valid JSON format, WITHOUT any EXPLANATIONS or INTRODUCTIONS. \
                Return ONLY the JSON object."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    data = response.json()
    if "choices" not in data or not data["choices"]:
        print(f"API response missing 'choices': {data}")
        return ""
    return data["choices"][0]["message"]["content"]

def generate_CV(prompt: str) -> str:
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are Pro in resume writing. Use your knowledge and what \
                will be given to produce a high-quality CV."},
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
    output = generate_key_words(prompt)
    # print(f"Output for CV was:\n{output}")
    try:
        cleaned = clean_json_markdown(output)
        print(f"Cleaned output: \n{cleaned}")
        if cleaned and cleaned[0] == '{':
            return json.loads(cleaned)
        else:
            print("Output does not start with '{', cannot parse as JSON.")
            print(f"Output for CV was:\n{output}")
            return {}
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
    output = generate_key_words(prompt)
    # print(f"Output for job offer was:\n{output}")
    try:
        cleaned = clean_json_markdown(output)
        print(f"Cleaned output: \n{cleaned}")
        if cleaned and cleaned[0] == '{':
            return json.loads(cleaned)
        else:
            print("Output does not start with '{', cannot parse as JSON.")
            print(f"Output for job offer was:\n{output}")
            return {}
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
    reformulated_cv_text = generate_CV(prompt)
    return reformulated_cv_text