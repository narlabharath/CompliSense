import os
from PyPDF2 import PdfReader
from vertexai.language_models import TextEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
PRODUCT_GUIDE_PATH = "client_inputs/product_guide.pdf"
EMBEDDING_MODEL = "textembedding-gecko@003"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
WINDOW_SIZE = 1  # Number of chunks before and after the best match

# === INIT EMBEDDING MODEL ===
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

# === UTILITY FUNCTIONS ===
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    return [
        " ".join(tokens[i:i+chunk_size])
        for i in range(0, len(tokens), chunk_size - overlap)
    ]

def embed(text):
    return embedding_model.get_embeddings([text])[0].values

def retrieve_product_with_window(product_name, pdf_path=PRODUCT_GUIDE_PATH, window_size=WINDOW_SIZE):
    print(f"🔍 Retrieving context window for: {product_name}")
    full_text = parse_pdf(pdf_path)
    chunks = chunk_text(full_text)

    product_vec = embed(product_name)
    scores = []

    for idx, chunk in enumerate(chunks):
        chunk_vec = embed(chunk)
        score = cosine_similarity([product_vec], [chunk_vec])[0][0]
        scores.append((score, idx))

    # Get the index with the highest score
    best_score, best_index = max(scores, key=lambda x: x[0])

    # Get window around best index
    start = max(0, best_index - window_size)
    end = min(len(chunks), best_index + window_size + 1)
    selected_chunks = chunks[start:end]

    print(f"✅ Retrieved chunks {start} to {end - 1} around best match (index {best_index}, score {best_score:.4f})")
    return "\n\n".join(selected_chunks)

# === EXAMPLE USAGE ===
# result = retrieve_product_with_window("FX Forward", window_size=2)
# print(result)

