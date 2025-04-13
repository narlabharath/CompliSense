"""
CompliSense Stage 1 - Knowledge Ingestion Script

This script:
- Parses PDFs from `data/input_documents/`
- Chunks them into manageable segments
- Uses Vertex AI embeddings (`textembedding-gecko@003`) to match content to known products/disclosures
- Uses Vertex AI LLM (`chat-bison`) to extract structured YAML knowledge
- Saves YAML to `normalized_knowledge/products/` and `.../disclosures/`
"""

import os
import yaml
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingModel, ChatModel
from datetime import datetime

# --- Configuration ---
INPUT_DIR = "data/input_documents"
PRODUCT_KB_DIR = "normalized_knowledge/products"
DISCLOSURE_KB_DIR = "normalized_knowledge/disclosures"
EMBEDDING_MODEL = "textembedding-gecko@003"
LLM_MODEL = "chat-bison"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MATCH_THRESHOLD = 0.75

# --- Vertex AI Init ---
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
chat_model = ChatModel.from_pretrained(LLM_MODEL)

def parse_pdf(file_path):
    print(f"📄 Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    return [
        " ".join(tokens[i:i+chunk_size])
        for i in range(0, len(tokens), chunk_size - overlap)
    ]

def get_known_entities(folder_path):
    return [
        f.replace(".yaml", "").replace("_", " ").title()
        for f in os.listdir(folder_path) if f.endswith(".yaml")
    ]

def match_entity(text_chunk, candidates, threshold=MATCH_THRESHOLD):
    chunk_vec = embedding_model.get_embeddings([text_chunk])[0].values
    best_match, best_score = None, 0.0

    for candidate in candidates:
        cand_vec = embedding_model.get_embeddings([candidate])[0].values
        score = cosine_similarity([chunk_vec], [cand_vec])[0][0]
        if score > best_score and score >= threshold:
            best_match = candidate
            best_score = score

    return best_match

def extract_yaml_from_llm(name, content, entity_type):
    print(f"🧠 Extracting structured {entity_type} for: {name}")
    prompt = f"""
You are a compliance assistant. Extract structured YAML data for the {entity_type} "{name}" from the content below:

{content}
"""
    chat = chat_model.start_chat()
    return chat.send_message(prompt).text  # Expecting LLM to return YAML directly

def save_yaml(yaml_text, output_folder, name):
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{name.lower().replace(' ', '_')}.yaml"
    path = os.path.join(output_folder, filename)
    try:
        parsed = yaml.safe_load(yaml_text)
        with open(path, "w") as f:
            yaml.dump(parsed, f, sort_keys=False)
        print(f"✅ Saved: {path}")
    except Exception as e:
        print(f"❌ Failed to parse/save YAML for {name}: {e}")

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    known_products = get_known_entities(PRODUCT_KB_DIR)
    known_disclosures = get_known_entities(DISCLOSURE_KB_DIR)

    for file in files:
        path = os.path.join(INPUT_DIR, file)
        chunks = chunk_text(parse_pdf(path))
        buffer = {"products": {}, "disclosures": {}}

        for chunk in chunks:
            product = match_entity(chunk, known_products)
            if product:
                buffer["products"].setdefault(product, "")
                buffer["products"][product] += chunk + "\n"
            else:
                disclosure = match_entity(chunk, known_disclosures)
                if disclosure:
                    buffer["disclosures"].setdefault(disclosure, "")
                    buffer["disclosures"][disclosure] += chunk + "\n"

        for name, content in buffer["products"].items():
            yaml_text = extract_yaml_from_llm(name, content, "product")
            save_yaml(yaml_text, PRODUCT_KB_DIR, name)

        for name, content in buffer["disclosures"].items():
            yaml_text = extract_yaml_from_llm(name, content, "disclosure")
            save_yaml(yaml_text, DISCLOSURE_KB_DIR, name)

    print("🎉 Ingestion complete.")

if __name__ == "__main__":
    main()
