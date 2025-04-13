# 🧠 CompliSense

**CompliSense** is an AI-powered compliance pipeline designed for regulated industries (like banking, insurance, and finance) that automates the extraction, structuring, and application of regulatory knowledge.

Built to support high-volume, high-stakes environments like customer call centers and trade confirmation desks, CompliSense transforms complex documents into structured, versioned compliance checklists — and applies them intelligently to post-call surveillance.

---

## 🚀 Key Features

- 🔍 **Stage 1: Knowledge Ingestion**
  - Parse PDFs (e.g. Order Management Guides, Risk Disclosures)
  - Classify content into **products** and **disclosures**
  - Extract checklist logic using Vertex AI (`chat-bison`)
  - Generate structured, version-controlled **YAML knowledge base**
  - Uses **Gecko embeddings** for semantic awareness (coming soon)

- ✅ **Stage 2: Post-Call Compliance (Coming Soon)**
  - Process call transcripts with timestamps and diarization
  - Detect which products were discussed, and when
  - Match against required disclosures and checklist items
  - Flag missing disclosures or gaps in compliance
  - Score calls based on **sales suitability logic**

---

## 🛡️ Architecture Overview

```
PDFs → Chunker → Classifier → Extractor → Knowledge Writer → YAML Knowledge Center
                                                 ↓
                                        (Optional) Registry
                                                 ↓
                                       Post-call Validator (Stage 2)
```

- **Knowledge Center** = source of truth (`normalized_knowledge/`)
- **No duplication**: everything is versioned and merged
- **Flexible**: config-driven merge behavior, registry auto-generation

---

## 🧐 Powered By Vertex AI

- 🤖 **LLM**: `chat-bison` or `chat@vertexAI`
- 🧠 **Embeddings**: `textembedding-gecko@003`
- 🗆 **PDF Parsing**: PyPDF2 / pdfplumber
- 🔧 Configurable via `config/system_config.yaml`

---

## 📁 Project Structure

```
compli-sense/
├── README.md
├── requirements.txt
├── config/
│   └── system_config.yaml
├── data/
│   └── input_documents/
├── normalized_knowledge/
│   ├── products/
│   └── disclosures/
├── stage1_ingestion/
│   ├── main.py
│   ├── components/
│   └── tests/
├── stage2_analysis/
│   ├── validator.py
│   └── tests/
└── utils/
```

---

## ⚙️ Configuration (`config/system_config.yaml`)

```yaml
merge_policy: add-only         # or: overwrite, conservative
strict_validation: false
auto_versioning: true
audit_enabled: false
registry_generation: true
```

---

## 📌 Roadmap

| Milestone        | Status       |
|------------------|--------------|
| Stage 1: Ingestion engine      | ✅ Complete (Core modules) |
| Vertex AI integration          | ✅ LLM + Embeddings ready |
| Stage 2: Compliance matching   | ➰ In development |
| YAML versioning + audit logs   | 🔚 Coming soon |
| Web dashboard (optional)       | 🔚 Coming soon |

---

## 👥 Use Cases

- 📻 Trade desk compliance (post-call surveillance)
- 💬 Real-time sales suitability monitoring
- 🗒️ Document-to-Checklist automation for QA
- 🤠 Knowledge management for regulated workflows

---

## 📣 Want to Contribute?

We welcome clean, testable, and modular contributions! Check out the `tests/` folder to see how to write new test cases for each module.

---

## 🏁 Getting Started (Coming Soon)

```bash
# Install deps
pip install -r requirements.txt

# Run Stage 1 ingestion
python stage1_ingestion/main.py --input ./data/input_documents/sample.pdf
```

---

## 🔒 License

MIT License — use freely, just don't break the rules it teaches 😉

---

> Built with ❤️ to make compliance less painful — and a lot more intelligent.

