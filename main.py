from src.rag import RAGMatcher
from src.pdf_reader import extract_text_from_pdf
import warnings
warnings.filterwarnings("ignore")

# Chargement des fichiers
# with open("data/cv.txt", "r", encoding="utf-8") as f:
#     cv_text = f.read()

cv_text = extract_text_from_pdf("data/cv.pdf")

with open("data/job.txt", "r", encoding="utf-8") as f:
    job_text = f.read()

# RAG Matching
matcher = RAGMatcher()
# score = matcher.compute_similarity(cv_text, job_text)
sections_scores = matcher.match_by_section(cv_text, job_text)

# print(f"Score de compatibilité : {score}/100")

if sections_scores['global'] < 80:
    print("⚠️ Le score est inférieur à 80. Le CV doit être amélioré.")
else:
    print("✅ Bon score, le CV est bien aligné avec l'offre.")

print("Scores par section :")
for section, section_score in sections_scores.items():
    print(f" - {section.capitalize()} : {section_score}/100")
    # print(f" - score global : {score}/100")

