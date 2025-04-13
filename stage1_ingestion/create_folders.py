import os

# Define required folders
required_folders = [
    "data/input_documents",
    "normalized_knowledge/products",
    "normalized_knowledge/disclosures"
]

for folder in required_folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

# Return created structure for confirmation
required_folders
