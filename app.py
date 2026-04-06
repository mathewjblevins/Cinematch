import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0a;
    color: #f0f0f0;
}

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem !important;
    letter-spacing: 4px;
    color: #E50914;
    margin-bottom: 0 !important;
}

.subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.stTextInput > div > div > input {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    padding: 0.75rem 1rem;
}

.stTextInput > div > div > input:focus {
    border-color: #E50914;
    box-shadow: 0 0 0 1px #E50914;
}

.stButton > button {
    background-color: #E50914;
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    text-transform: uppercase;
    transition: background 0.2s;
}

.stButton > button:hover {
    background-color: #b20610;
}

.movie-card {
    background: #141414;
    border-radius: 6px;
    overflow: hidden;
    transition: transform 0.2s;
    height: 100%;
}

.movie-card:hover {
    transform: scale(1.03);
}

.movie-card img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
}

.card-body {
    padding: 0.75rem;
}

.card-title {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    color: #f0f0f0;
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-meta {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 0.75rem;
    color: #888;
}

.card-rating {
    color: #f5c518;
    font-size: 0.8rem;
    font-weight: 500;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 2rem 0;
}

.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 3px;
    color: #888;
    margin-bottom: 1rem;
}

.match-badge {
    display: inline-block;
    background: #E50914;
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.stAlert {
    background-color: #1a1a1a;
    border: 1px solid #333;
    color: #f0f0f0;
}

footer {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #444;
    text-align: center;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)

# ── TMDB API ──────────────────────────────────────────────────────────────────
API_KEY = st.secrets.get("TMDB_API_KEY", "5e547a886033acdf0c838a0574b65068")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/500x750/141414/444?text=No+Image"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movies(pages=10):
    """Fetch popular + top-rated movies from TMDB."""
    movies = {}
    genre_map = get_genre_map()

    for page in range(1, pages + 1):
        for endpoint in ["movie/popular", "movie/top_rated"]:
            r = requests.get(
                f"{BASE_URL}/{endpoint}",
                params={"api_key": API_KEY, "language": "en-US", "page": page},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            for m in r.json().get("results", []):
                mid = m["id"]
                if mid not in movies:
                    genres = " ".join(
                        genre_map.get(g, "") for g in m.get("genre_ids", [])
                    )
                    movies[mid] = {
                        "id": mid,
                        "title": m.get("title", ""),
                        "overview": m.get("overview", ""),
                        "genres": genres,
                        "poster_path": m.get("poster_path", ""),
                        "vote_average": m.get("vote_average", 0),
                        "release_date": m.get("release_date", ""),
                        "popularity": m.get("popularity", 0),
                    }
    return pd.DataFrame(list(movies.values()))


@st.cache_data(show_spinner=False, ttl=86400)
def get_genre_map():
    r = requests.get(
        f"{BASE_URL}/genre/movie/list",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=10,
    )
    if r.status_code == 200:
        return {g["id"]: g["name"] for g in r.json().get("genres", [])}
    return {}


@st.cache_data(show_spinner=False)
def build_similarity_matrix(df):
    """TF-IDF on combined text features, return cosine similarity matrix."""
    df = df.copy()
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    # Weight genres more by repeating them
    df["features"] = df["overview"] + " " + df["genres"] * 3

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["features"])
    return cosine_similarity(matrix)


def get_recommendations(title, df, sim_matrix, n=10):
    """Return top-n similar movies for a given title."""
    title_lower = title.lower().strip()
    matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]
    if matches.empty:
        return None, None

    # Pick the most popular match
    idx = matches.sort_values("popularity", ascending=False).index[0]
    pos = df.index.get_loc(idx)

    scores = list(enumerate(sim_matrix[pos]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Skip the movie itself
    scores = [s for s in scores if s[0] != pos][:n]

    result_indices = [s[0] for s in scores]
    result_scores = [s[1] for s in scores]
    return df.iloc[result_indices], result_scores


def poster_url(path):
    return f"{IMG_BASE}{path}" if path else PLACEHOLDER


def year(date_str):
    return date_str[:4] if date_str and len(date_str) >= 4 else "N/A"


def render_movie_card(movie, score=None):
    badge = f'<div class="match-badge">Match</div>' if score else ""
    rating_stars = "★" * round(movie["vote_average"] / 2)
    html = f"""
    <div class="movie-card">
        <img src="{poster_url(movie['poster_path'])}" alt="{movie['title']}" loading="lazy"/>
        <div class="card-body">
            {badge}
            <div class="card-title">{movie['title']}</div>
            <div class="card-meta">
                <span class="card-rating">{rating_stars}</span>
                {movie['vote_average']:.1f} &nbsp;·&nbsp; {year(movie['release_date'])}
            </div>
        </div>
    </div>
    """
    return html


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown('<h1>CINEMATCH</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Content-Based Movie Recommendations</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading movie database..."):
    df = fetch_movies(pages=10)
    df = df.reset_index(drop=True)
    sim_matrix = build_similarity_matrix(df)

# Search bar
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "",
        placeholder="Enter a movie title (e.g. The Dark Knight, Inception, Interstellar...)",
        label_visibility="collapsed",
    )
with col2:
    search_btn = st.button("Find Matches", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Results
if query and (search_btn or query):
    recs, scores = get_recommendations(query, df, sim_matrix, n=10)

    if recs is None:
        st.warning(f'No movie found matching "{query}". Try a different title.')
    else:
        # Find the matched movie
        title_lower = query.lower().strip()
        matched = df[df["title"].str.lower().str.contains(title_lower, na=False)]
        matched_movie = matched.sort_values("popularity", ascending=False).iloc[0]

        st.markdown(f'<p class="section-label">Because you searched: {matched_movie["title"]}</p>', unsafe_allow_html=True)

        # Show matched movie
        col_poster, col_info = st.columns([1, 3])
        with col_poster:
            st.markdown(
                f'<img src="{poster_url(matched_movie["poster_path"])}" style="width:100%;border-radius:6px;" />',
                unsafe_allow_html=True,
            )
        with col_info:
            st.markdown(f"### {matched_movie['title']} ({year(matched_movie['release_date'])})")
            st.markdown(f"**Rating:** {'★' * round(matched_movie['vote_average']/2)} {matched_movie['vote_average']:.1f}/10")
            st.markdown(f"**Genres:** {matched_movie['genres']}")
            st.markdown(f"**Overview:** {matched_movie['overview']}")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Top Recommendations</p>', unsafe_allow_html=True)

        # Render recommendation grid
        cols = st.columns(5)
        for i, (_, movie) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.markdown(render_movie_card(movie, scores[i]), unsafe_allow_html=True)

else:
    # Default: show trending
    st.markdown('<p class="section-label">Trending Now</p>', unsafe_allow_html=True)
    trending = df.sort_values("popularity", ascending=False).head(10)
    cols = st.columns(5)
    for i, (_, movie) in enumerate(trending.iterrows()):
        with cols[i % 5]:
            st.markdown(render_movie_card(movie), unsafe_allow_html=True)

st.markdown("""
<footer>
    CineMatch · Built with TMDB API · ML Course Project · Valparaiso University 2025
</footer>
""", unsafe_allow_html=True)
