## 🔍 Information Retrieval System (TF-IDF CLI)


A modular command-line application for searching and ranking text documents using TF-IDF and cosine similarity. Built in pure Python with no external dependencies beyond the standard library and NumPy.


---
## 📁 Directory Structure

```
ir_system/
├── data/       # Folder containing .txt documents to be indexed
│   ├── doc1.txt
│   ├── doc2.txt
│   ├── doc3.txt
│   ├── doc4.txt
│   ├── doc5.txt
│   └── README          # Optional legacy readme
├── Scripts/            # Core Python modules
│   ├── ir_core.py    # TF-IDF engine and cosine similarity logic
│   ├── search_cli.py   # Interactive command-line interface
│   └── __pycache__/    # Python bytecode cache
└── README.md           # Project documentation

```

---

## 🚀 Features

- ✅ Load and index `.txt` documents from the `data/` folder
- ✅ Compute TF-IDF vectors and cosine similarity
- ✅ Search interactively via CLI
- ✅ View document content and system statistics
- ✅ Fuzzy matching for mistyped document IDs
- ✅ Modular design for easy extension (e.g., semantic search, GUI)

---

## 🧪 How to Run

### 1. Add `.txt` files to the `data/` folder

Each file should contain plain text. Example filenames:
- `doc1.txt`
- `doc2.txt`

### 2. Launch the CLI

```bash
cd Scripts
python search_cli.py
```

### 3. Use the CLI commands

```plaintext
Search> stats           # Show system statistics
Search> list            # List all documents
Search> view doc1       # View content of a document
Search> machine learning algorithms  # Perform a search
Search> quit            # Exit the CLI
```

---

## 📊 System Statistics

After indexing, the system displays:
- Total number of documents
- Vocabulary size
- Average document length
- Index status (built or not)

---

## 🧠 How It Works

- Documents are tokenized and term frequencies are calculated.
- IDF scores are computed across the corpus.
- Each document is vectorized using TF-IDF.
- Queries are vectorized similarly and compared using cosine similarity.
- Top-k results are returned with relevance scores and previews.

---

## 🛠️ Requirements

- Python 3.7+
- NumPy

Install NumPy if needed:

```bash
pip install numpy
```

---

## 📌 Notes

- All `.txt` files must be placed in the `data/` folder before running the CLI.
- The system is designed for extensibility—semantic search and GUI integration can be added easily.
---
