from llama_cpp import Llama

llm = Llama(
    model_path="models/hermes-llama2-13b.gguf",
    n_gpu_layers=12,  # pour tout décharger sur GPU
    verbose=True
)
print("Modèle chargé !")