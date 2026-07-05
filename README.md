 #PaperLens

Find the right research paper in seconds — not by keywords, but by meaning.



---

## What is this?

You type a question like *"deep learning for medical image analysis"* — PaperLens searches thousands of ML papers, ranks them by true relevance (not just word overlap), and hands you back a short summary and key topics for each, so you can decide what's worth reading in seconds.

## The Pipeline

**Search → Rerank → Summarize → Extract Keywords**

| Stage | What happens | Model used |
|---|---|---|
|  Retrieve | Embed your query and find the closest papers in vector space | `all-MiniLM-L6-v2` + FAISS |
|  Rerank | Re-score the top candidates for true relevance | `ms-marco-MiniLM-L-6-v2` |
|  Summarize | Compress each abstract into a few readable sentences | `bart-large-cnn` |
|  Extract keywords | Pull out the key phrases behind each result | KeyBERT |

## Try It

```python
search_rerank_and_summarize("deep learning for medical image analysis", k=20, top_n=5)
```

```
Cross-encoder score: 8.24
Title: A Perspective on Deep Imaging
Summary: The combination of tomographic imaging and deep learning promises
         to empower not only image analysis but also image reconstruction...
Keywords: tomographic imaging, medical imaging, deep learning
```

## Quick Start

```bash
pip install transformers==4.46.3 sentence-transformers faiss-cpu keybert==0.8.5 datasets pandas
```

```python
from datasets import load_dataset
import pandas as pd, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from keybert import KeyBERT
from transformers import pipeline

# Load 15k ML papers
df = pd.DataFrame(load_dataset("CShorten/ML-ArXiv-Papers", split="train"))[["title", "abstract"]].head(15000)
df["paper_text"] = df["title"] + " " + df["abstract"]

# Load models
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT(model)

# Index papers
embedding = model.encode(df["paper_text"].tolist(), show_progress_bar=True)
faiss.normalize_L2(embedding)
index = faiss.IndexFlatIP(384)
index.add(embedding)
```

## Why Two-Stage Search?

Vector search alone (bi-encoder) is fast but approximate — it compares query and paper embeddings *separately*, missing fine-grained interactions between them. So PaperLens retrieves a wide candidate pool first (fast), then re-scores that shortlist with a cross-encoder that reads the query and paper *together* — the same "retrieve-then-rerank" pattern used in real-world search engines.

## Dataset

[CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) — titles and abstracts of machine learning papers from arXiv.

## What's Next

- Cluster papers into topics automatically
- Ask questions and get answers grounded in retrieved papers (RAG)
- A simple web demo (Gradio/Streamlit)
- "Papers like this one" recommendations

## License

MIT
