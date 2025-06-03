# ⚖️ Legal Case Classifier & Summariser

#<p align="center">
# <img src="https://raw.githubusercontent.com/github/explore/main/topics/law/law.png" width="120" alt="Legal NLP Icon"/>
#</p>

An end-to-end GPU-accelerated pipeline to scrape, classify, and summarise Indian legal case documents. Built for scale, tuned for precision.

---

## 🧠 Pipeline Overview

1. **Web Scraping** – Extracts case data from official court sources into CSV (~250MB).
2. **Preprocessing** – Lowercasing, stop word removal, lemmatization, and n-gram generation.
3. **Classification** – Heuristic rule-based prediction using manually vectorized keywords.
4. **Summarization** – Concise, 100-word summaries using transformer models (T5 / BART) with professional legal tone.

---

## ✨ Features

- 📄 Scrapes and structures raw legal text from court portals.
- 🧹 NLP preprocessing using SpaCy and NLTK.
- 🧠 Heuristic classification of content (issues, petitions, conclusions, arguments).
- 📝 T5/BART-based summarization via HuggingFace or local inference.
- ⚡ CUDA-enabled for fast training and inference.
- 🧩 Modular pipeline design.

---

## 📈 Future Roadmap

- 🧠 Build a custom summarization model replicating T5 architecture.
- 📦 Scale dataset up to 1TB with broader legal domain coverage.
- ⚙️ Integrate parallel computing for large-scale training.
- 🌐 Deploy complete webapp with case upload, live classification, and summarization.
- 🧪 Fine-tune on domain-specific legal jargon for Indian courts.

---
