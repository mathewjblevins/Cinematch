CineMatch

A content-based movie recommendation engine built with Python, scikit-learn, and Streamlit.

How It Works

CineMatch uses **content-based filtering** to recommend movies similar to one you already like:

1. **Data** — Fetches movie data (overviews, genres, ratings, posters) live from the TMDB API
2. **Vectorization** — Applies TF-IDF vectorization to each movie's overview and genre tags
3. **Similarity** — Computes cosine similarity between all movie vectors
4. **Recommendations** — Returns the top 10 most similar movies ranked by similarity score

## Why Content-Based Filtering?

- No cold start problem, so it works without any user history
- Explainable so the recommendations are based on movie attributes, not black-box user patterns
- Live data always up to date via TMDB API

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn (TF-IDF + Cosine Similarity) |
| Data | TMDB API |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch
pip install -r requirements.txt
```

Add your TMDB API key to `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_key_here"
```

Run the app:
```bash
streamlit run app.py
```

## Course

Machine Learning & Data Mining — Valparaiso University, Spring 2025
