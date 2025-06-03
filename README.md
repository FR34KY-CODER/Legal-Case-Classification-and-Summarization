# ⚖️ Legal Case Classifier & Summariser

<p align="center">
<img src="https://github.com/FR34KY-CODER/Legal-Case-Classification-and-Summarization/blob/main/image.png" alt="Legal NLP Icon"/>
</p>

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
<img src="https://private-user-images.githubusercontent.com/74038190/240885606-f606466f-4cc9-4cb1-8ad6-80a7eeea9e7e.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDg5NDczMTksIm5iZiI6MTc0ODk0NzAxOSwicGF0aCI6Ii83NDAzODE5MC8yNDA4ODU2MDYtZjYwNjQ2NmYtNGNjOS00Y2IxLThhZDYtODBhN2VlZWE5ZTdlLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjAzVDEwMzY1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWE1OTA1ZTgxMzA3NTMzMTAwMzM4NzZlMmIxYjg1Mzk1MDBhMzg2NjU1NjJhZmJhY2QwMzA3MWNiMDVhMzg5ZDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.U59N_E7Q8BHIFIt148-3nmLDWS60JpvM9Bh46iio0IY">
