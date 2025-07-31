from sklearn.metrics.pairwise import cosine_similarity
from .embedder import Embedder
from .section_extractor import extract_sections

class RAGMatcher:
    def __init__(self):
        self.embedder = Embedder()

    def compute_similarity(self, cv_text: str, job_text: str) -> float:
        cv_vec = self.embedder.embed_text(cv_text)
        job_vec = self.embedder.embed_text(job_text)
        similarity = cosine_similarity([cv_vec], [job_vec])[0][0]
        return round(similarity * 100, 2)
    
    def match_by_section(self, cv_text: str, job_text: str) -> dict:
        cv_sections = extract_sections(cv_text)
        job_sections = extract_sections(job_text)

        print("\n🧾 Sections extraites du CV :")
        for key, val in cv_sections.items():
            print(f"- {key}: {val[:5]}")

        print("\n📄 Sections extraites de l'offre :")
        for key, val in job_sections.items():
            print(f"- {key}: {val[:5]}")
            
        scores = {}

        for section in ["skills", "experience", "keywords"]:
            cv_part = " ".join(cv_sections[section])
            job_part = " ".join(job_sections[section])
            if cv_part and job_part:
                scores[section] = round(self.compute_similarity(cv_part, job_part), 2)
            else:
                scores[section] = 0.0

        # Pondération : par défaut on considère skills > experience > keywords
        global_score = round(
            0.5 * scores["skills"] +
            0.3 * scores["experience"] +
            0.2 * scores["keywords"],
            2
        )

        scores["global"] = global_score
        return scores