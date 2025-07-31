import re

def extract_sections(text: str) -> dict:
    sections = {
        "skills": [],
        "experience": [],
        "keywords": []
    }

    # Extraire la section compétences (mot-clé : compétences ou skills)
    skills_match = re.search(r"(compétences|skills|technical skills)[\s:\-]*([\s\S]{0,500})", text, re.IGNORECASE)
    if skills_match:
        raw_skills = skills_match.group(2)
        skills = re.split(r",|\n", raw_skills)
        sections["skills"] = [s.strip() for s in skills if s.strip()]

    # Extraire les expériences
    experience_match = re.search(r"(expérience|experience|professional experience)[\s:\-]*([\s\S]{0,1000})", text, re.IGNORECASE)
    if experience_match:
        raw_experience = experience_match.group(2)
        lines = raw_experience.split("\n")
        sections["experience"] = [line.strip() for line in lines if line.strip()]

    # Mots-clés potentiels (liste de mots fréquents)
    words = re.findall(r"\b[a-zA-Zéèàêâç\-]{3,}\b", text.lower())
    keywords = list(set(words))
    sections["keywords"] = keywords

    return sections
