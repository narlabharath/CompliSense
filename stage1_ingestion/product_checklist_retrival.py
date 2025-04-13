import os
from PyPDF2 import PdfReader
from vertexai.language_models import TextEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
PRODUCT_GUIDE_PATH = "client_inputs/product_guide.pdf"
EMBEDDING_MODEL_NAME = "textembedding-gecko@003"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
WINDOW_SIZE = 1  # Number of chunks before and after the best match

# === INIT EMBEDDING MODEL ===
print("🔁 Loading embedding model...")
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

# === UTILITY FUNCTIONS ===
def parse_pdf(file_path):
    print(f"📄 Reading and extracting text from PDF: {file_path}")
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    print(f"✅ Extracted {len(text)} characters of raw text from PDF.")
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    print("✂️ Chunking text...")
    tokens = text.split()
    chunks = [
        " ".join(tokens[i:i+chunk_size])
        for i in range(0, len(tokens), chunk_size - overlap)
    ]
    print(f"✅ Created {len(chunks)} overlapping chunks.")
    return chunks

def embed(texts):
    print(f"🧠 Embedding {len(texts)} text blocks...")
    return embedding_model.embed_documents(texts)

def retrieve_product_with_window(product_name, pdf_path=PRODUCT_GUIDE_PATH, window_size=WINDOW_SIZE):
    print(f"\n🚀 Retrieving product section for: '{product_name}' with context window = {window_size}")
    
    # Step 1: Read and chunk the PDF
    full_text = parse_pdf(pdf_path)
    chunks = chunk_text(full_text)

    # Step 2: Embed chunks and product name
    chunk_embeddings = embed(chunks)
    product_embedding = embedding_model.embed_documents([product_name])[0]

    # Step 3: Score each chunk by similarity
    print("🔍 Calculating cosine similarities...")
    scores = []
    for i, chunk_vec in enumerate(chunk_embeddings):
        score = cosine_similarity(
            [product_embedding.values],
            [chunk_vec.values]
        )[0][0]
        scores.append((score, i))
        print(f"   - Chunk {i:02}: Score = {score:.4f}")

    # Step 4: Find the best match and get window
    best_score, best_index = max(scores, key=lambda x: x[0])
    print(f"\n🏆 Best match: Chunk {best_index} with score {best_score:.4f}")

    start = max(0, best_index - window_size)
    end = min(len(chunks), best_index + window_size + 1)
    selected_chunks = chunks[start:end]

    print(f"📦 Returning chunks {start} to {end - 1} (total {len(selected_chunks)} chunks)")
    return "\n\n".join(selected_chunks)

# === EXAMPLE USAGE ===
# result = retrieve_product_with_window("FX Forward", window_size=2)
# print(result)

