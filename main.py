from src.rag import RAGMatcher
from src.pdf_reader import extract_text_from_pdf
from src.llm_online import extract_cv_info, extract_job_info, reformulate_cv_for_job
import os
import markdown
import pdfkit
import time
import warnings
warnings.filterwarnings("ignore")

def save_markdown_to_pdf(markdown_text: str, output_path: str):


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = markdown.markdown(markdown_text)
    pdfkit.from_string(html, output_path)

def main():
    cv_text = extract_text_from_pdf("data/cv.pdf")
    parsed_cv = extract_cv_info(cv_text)

    with open("data/job.txt", "r", encoding="utf-8") as f:
        job_text = f.read()
    parsed_job = extract_job_info(job_text)
    time.sleep(3)  # Just to ensure the loading animation has time to start

    matcher = RAGMatcher()

    def evaluate(cv_structured, job_structured):
        scores = matcher.match_by_section(cv_structured, job_structured)
        print("\nScores par section :")
        for section, score in scores.items():
            print(f" - {section.capitalize()} : {score}/100")
        return scores

    print("✅ CV structuré :")
    print(parsed_cv)

    print("✅ Offre structuré :")
    print(parsed_job)

    time.sleep(3)  # Just to ensure the loading animation has time to start
    sections_scores = evaluate(parsed_cv, parsed_job)

    while sections_scores['global'] < 80:
        print("⚠️ Le score est inférieur à 80. Le CV doit être amélioré.")
        choice = input("Voulez-vous reformuler votre CV pour mieux matcher l'offre ? (y(es)/n(o)) : ").strip().lower()
        if choice in ['y', 'yes']:
            # Appelle ta fonction de reformulation ici
            reformulated_cv_text = reformulate_cv_for_job(cv_text, job_text, parsed_job)
            print("\n📝 CV reformulé pour correspondre à l'offre :\n")
            print(reformulated_cv_text)

            save_markdown_to_pdf(reformulated_cv_text, "output/cv_reformulated.pdf")
            print("✅ CV reformulé sauvegardé en PDF : output/cv_reformulated.pdf")

            # Re-extraction et re-évaluation
            parsed_cv = extract_cv_info(reformulated_cv_text)  # Si extract_cv_info supporte str, sinon adapter
            sections_scores = evaluate(parsed_cv, parsed_job)
        elif choice in ['n', 'no']:
            print("OK, pas de reformulation.")
            break
        else:
            print("Choix invalide. Veuillez répondre par 'y' ou 'n'.")

    if sections_scores['global'] >= 80:
        print("✅ Bon score, le CV est bien aligné avec l'offre.")

if __name__ == "__main__":
    main()
