# WebRAG: Enhancing Retrieval-Augmented Generation with Web Link Structure

This repository contains the implementation of **WebRAG**, a Web-aware RAG system that enhances multi-hop question answering by integrating hyperlink-based document graph structures into the retrieval process. By combining traditional text-based similarity with web document interconnections, WebRAG significantly improves answer accuracy for complex queries across multiple documents.

## 🔍 Overview

Traditional Retrieval-Augmented Generation (RAG) systems retrieve passages based solely on semantic or lexical similarity. However, they often struggle with **multi-hop reasoning**, especially when information is spread across different documents.

**WebRAG** introduces a novel enhancement:

- Constructs a hyperlink graph between documents (using links in Web pages).
- Expands context retrieval by exploring the graph neighborhood of relevant pages.
- Merges retrieved text from both RAG and WebGraph modules to generate more accurate and context-rich answers.

## 📊 Key Features

- Combines RAG with hyperlink-based Web graphs.
- Designed for **multi-hop question answering** tasks.
- Evaluated on benchmark datasets: **2WikiMultiHopQA** and **HotpotQA**.
- Outperforms standard RAG and shows strong performance compared to more complex graph-based methods.

## 📁 Project Structure

```bash
├── Data/
│   ├── 2WikiMultihopQA/           # 2WikiMultiHopQA dataset files
│   └── HotpotQA/                  # HotpotQA dataset files
│
├── RAG/                           # Core RAG logic and helper modules
│   ├── __import__.py              # Import handling (likely placeholder)
│   ├── __init__.py                # Package initializer
│   ├── func.py                    # Utility functions
│   ├── rag.py                     # Main RAG implementation
│   ├── relatedData.py             # Handles related data augmentation
│   └── SystemPrompts/            # System prompt templates
│       ├── prompt-rag.txt
│       ├── prompt-rag&related_data.txt
│       ├── prompt-rag+related_data.txt
│       └── prompt-related_data.txt
│
├── eval.py                        # Evaluation logic (e.g., EM/F1 computation)
├── main.py                        # Script to run the pipeline
├── main.ipynb                     # Notebook for interactive exploration
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore rules
```

## 🧪 Datasets

We use:

- **2WikiMultiHopQA**: Focused on compositional reasoning over Wikipedia.
- **HotpotQA**: Natural multi-hop questions with supporting paragraphs.

> 📌 Note: Our experiments are conducted on Wikipedia-based datasets, which are highly structured. Future work will explore broader Web content such as blogs, forums, and enterprise data.

## ⚙️ Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/WebRAG.git
cd WebRAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the datasets (or place them in the `data/` directory).

## 🚀 Usage

Run the main script with:

```bash
python main.py
```

## 📈 Results

WebRAG significantly boosts EM and F1 scores over baseline RAG, particularly on the HotpotQA dataset where it achieves:

- **EM**: 80.0%
- **F1**: 82.2%

## 📚 Citation

If you use WebRAG in your research, please cite our work:

```bibtex
@inproceedings{yourname2025webrag,
  title={Enhancing Retrieval-Augmented Generation with Document Link Structure for Multi-hop Web Question Answering},
  author={Your Name et al.},
  booktitle={Proceedings of CASCON},
  year={2025}
}
```

## 🧠 Future Work

- Extend evaluation to more diverse, unstructured web sources.
- Improve graph construction beyond hyperlinks (e.g., semantic linking).
- Explore hybrid architectures combining different graph types.
