CineMatch

A content-based movie recommendation engine built with Python and Streamlit.

How It Works

CineMatch uses **content-based filtering** to recommend movies similar to one you already like:

1. **Data** Fetches movie data (overviews, genres, ratings, posters) live from the TMDB API
2. **Vectorization** Applies TF-IDF vectorization to each movie's overview and genre tags
3. **Similarity** Computes cosine similarity between all movie vectors
4. **Recommendations** Returns the top 10 most similar movies ranked by similarity score

### Local Setup

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
