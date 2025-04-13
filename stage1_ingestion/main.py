import os
import json
import yaml
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingModel, ChatModel

# === CONFIG ===
PRODUCT_PDF_PATH = "client_inputs/product_guide.pdf"
DISCLOSURE_PDF_PATH = "client_inputs/risk_disclosures.pdf"
FLAGS_PATH = "client_inputs/disclosure_flags.json"
OUTPUT_PRODUCT_DIR = "normalized_knowledge/products"
OUTPUT_DISCLOSURE_DIR = "normalized_knowledge/disclosures"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MATCH_THRESHOLD = 0.75

# === INIT MODELS ===
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
chat_model = ChatModel.from_pretrained("chat-bison")

# === UTILITIES ===
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size - overlap)]

def extract_entities_from_chunk(chunk, entity_type):
    prompt = f"""
You are a compliance assistant. Extract structured YAML for any {entity_type}s mentioned below.
Only include YAML. If nothing is found, return nothing.

{chunk}
"""
    chat = chat_model.start_chat()
    return chat.send_message(prompt).text.strip()

def parse_yaml_blocks(yaml_text):
    blocks = yaml_text.strip().split("---")
    parsed = []
    for block in blocks:
        try:
            parsed.append(yaml.safe_load(block))
        except Exception:
            continue
    return [b for b in parsed if b]

def save_yaml(data, folder, name):
    os.makedirs(folder, exist_ok=True)
    file_name = f"{name.lower().replace(' ', '_')}.yaml"
    with open(os.path.join(folder, file_name), "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"✅ Saved: {folder}/{file_name}")

def load_flags(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"required_disclosures": []}

# === PIPELINE ===
def run_entity_extraction_pipeline():
    print("🚀 Starting entity extraction pipeline...")

    # Step 1: Extract from product guide
    product_text = parse_pdf(PRODUCT_PDF_PATH)
    product_chunks = chunk_text(product_text)

    product_entities = []
    for chunk in product_chunks:
        response = extract_entities_from_chunk(chunk, "product")
        if response:
            product_entities.extend(parse_yaml_blocks(response))

    for entity in product_entities:
        name = entity.get("product_name", "unknown_product")
        save_yaml(entity, OUTPUT_PRODUCT_DIR, name)

    # Step 2: Extract from disclosure guide
    disclosure_text = parse_pdf(DISCLOSURE_PDF_PATH)
    disclosure_chunks = chunk_text(disclosure_text)

    disclosure_entities = []
    for chunk in disclosure_chunks:
        response = extract_entities_from_chunk(chunk, "disclosure")
        if response:
            disclosure_entities.extend(parse_yaml_blocks(response))

    for entity in disclosure_entities:
        name = entity.get("disclosure_name", "unknown_disclosure")
        save_yaml(entity, OUTPUT_DISCLOSURE_DIR, name)

    # Step 3: Optional - Print disclosure flag status
    flags = load_flags(FLAGS_PATH)
    if flags["required_disclosures"]:
        print("\n📋 Required Disclosures from Flags:")
        for d in flags["required_disclosures"]:
            print(f" - {d}")

    print("\n✅ Entity extraction complete. Structured YAML files are ready.")

# === RUN ===
run_entity_extraction_pipeline()
