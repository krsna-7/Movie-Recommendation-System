# 🎬 Movie Recommendation System

A Streamlit app that recommends movies using TMDb data, Sentence-BERT embeddings, and fuzzy search.

## Features
- Fetches popular movies from the TMDb API
- Intelligent fuzzy search with RapidFuzz
- Filter movies by rating and release year
- Displays movie details: poster, overview, release info, and trailer (YouTube)
- Sentence-BERT based movie recommendations using cosine similarity
- Interactive UI with caching for improved performance

## Tech Stack
Python, Streamlit, Pandas, NumPy, Sentence-BERT, Scikit-learn, RapidFuzz, Requests, TMDb API

## Usage
```bash
pip install -r requirements.txt
streamlit run app.py
