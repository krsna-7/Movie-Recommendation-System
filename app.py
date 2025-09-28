import torch, types
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

import streamlit as st
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide", page_icon="🎥")

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import time
from rapidfuzz import process, fuzz

TMDB_API_KEY = "4fe4c1131cd6a5f38bcb48d7608cac02"
TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0ZmU0YzExMzFjZDZhNWYzOGJjYjQ4ZDc2MDhjYWMwMiIsIm5iZiI6MTc0ODc1NzE5NS4zMDQ5OTk4LCJzdWIiOiI2ODNiZWFjYmM4ZmE0MGIyODFmMmIxODMiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.4HTkV7HCpe5UxLGxh8VlSED4yPEvDl7yIvrcWP4u0v8"

CACHE_FILE = "cached_movies.pkl"
CACHE_EXPIRY_SECONDS = 60 * 60 * 12

def is_cache_valid(file_path, expiry_seconds):
    if not os.path.exists(file_path):
        return False
    last_mod_time = os.path.getmtime(file_path)
    return (time.time() - last_mod_time) <= expiry_seconds

@st.cache_resource(show_spinner=False)
def load_sentence_bert_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda x: hash(x.to_csv())})
def generate_embeddings(texts):
    model = load_sentence_bert_model()
    return model.encode(texts, show_progress_bar=True)

def save_cache(df, path):
    df.to_pickle(path)

def load_cache(path):
    return pd.read_pickle(path)

@st.cache_data(show_spinner=False)
def fetch_movies(pages=3):
    if is_cache_valid(CACHE_FILE, CACHE_EXPIRY_SECONDS):
        return load_cache(CACHE_FILE)

    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "Content-Type": "application/json;charset=utf-8"
    }
    movies = []
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?page={page}&api_key={TMDB_API_KEY}"
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            st.error(f"❌ TMDb API Error {res.status_code}: {res.text}")
            return pd.DataFrame()
        for movie in res.json().get("results", []):
            release_date = movie.get('release_date', '')
            release_year = int(release_date[:4]) if release_date and release_date[:4].isdigit() else 0
            movies.append({
                'id': movie.get('id'),
                'title': movie.get('title', 'N/A'),
                'overview': movie.get('overview') or "No description available.",
                'release_date': release_date if release_date else 'N/A',
                'release_year': release_year,
                'vote_average': movie.get('vote_average', 0),
                'poster_path': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else ""
            })
    df = pd.DataFrame(movies)
    df['display'] = df['title'] + " (" + df['release_year'].astype(str) + ")"
    save_cache(df, CACHE_FILE)
    return df

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "Content-Type": "application/json;charset=utf-8"
    }
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None
    videos = res.json().get('results', [])
    for video in videos:
        if video['site'].lower() == 'youtube' and video['type'].lower() == 'trailer':
            return f"https://www.youtube.com/embed/{video['key']}"
    return None

def recommend(selected_display, movies_df, embeddings, top_n=5, min_rating=0, min_year=0):
    idx = movies_df[movies_df['display'] == selected_display].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    if len(embeddings) == 0 or idx >= len(embeddings):
        return pd.DataFrame()
    similarity_scores = cosine_similarity([embeddings[idx]], embeddings)[0]
    scores_with_index = list(enumerate(similarity_scores))
    filtered = [(i, score) for i, score in scores_with_index
                if movies_df.iloc[i]['vote_average'] >= min_rating and movies_df.iloc[i]['release_year'] >= min_year]
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
    filtered = [i for i, score in filtered if i != idx]
    top_indices = filtered[:top_n]
    return movies_df.iloc[top_indices]

def init_session_state():
    if 'selected_movie' not in st.session_state:
        st.session_state['selected_movie'] = None
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = pd.DataFrame()

def main():
    st.markdown(
        """
        <style>
        .main-header {
            text-align: center; 
            color: #F63366; 
            font-weight: 800;
            margin-bottom: 0.25rem;
            font-size: 3rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sub-header {
            text-align: center; 
            color: gray; 
            margin-top: 0; 
            margin-bottom: 1.5rem;
            font-size: 1.25rem;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='main-header'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='sub-header'>Powered by TMDb + Sentence-BERT + RapidFuzz</h4>", unsafe_allow_html=True)
    st.divider()

    init_session_state()

    with st.spinner("📦 Fetching movies from TMDb..."):
        movies_df = fetch_movies()

    if movies_df.empty:
        st.warning("⚠️ No movies available to display.")
        return

    with st.sidebar:
        st.header("🔧 Filters & Controls")
        min_rating = st.slider("Minimum rating ⭐", 0.0, 10.0, 0.0, 0.1)
        min_year = st.slider("Release year from 📅", 1900, 2025, 2000)
        search_movie = st.text_input("Search for a movie 🎞️")

        filtered_movies = movies_df[
            (movies_df['vote_average'] >= min_rating) & 
            (movies_df['release_year'] >= min_year)
        ]

        if search_movie:
            matches = process.extract(search_movie, filtered_movies['display'].tolist(), scorer=fuzz.WRatio, limit=100)
            filtered_titles = [match[0] for match in matches if match[1] > 60]
        else:
            filtered_titles = filtered_movies['display'].tolist()

        if not filtered_titles:
            filtered_titles = ["No movies found"]

        if st.session_state['selected_movie'] not in filtered_titles:
            st.session_state['selected_movie'] = filtered_titles[0]

        selected_movie = st.selectbox(
            "🎥 Choose a Movie",
            sorted(filtered_titles),
            index=sorted(filtered_titles).index(st.session_state['selected_movie']) if st.session_state['selected_movie'] in filtered_titles else 0,
            key="movie_select_box"
        )

        st.session_state['selected_movie'] = selected_movie

        if st.button("🎲 Pick Random Movie"):
            if filtered_titles and filtered_titles[0] != "No movies found":
                st.session_state['selected_movie'] = random.choice(filtered_titles)
                st.success(f"Randomly selected: {st.session_state['selected_movie']}")
                st.rerun()
            else:
                st.warning("No movies to select from.")

        top_n = st.slider("Number of Recommendations", 3, 10, 5)
        run_reco = st.button("🔍 Get Recommendations")

    if selected_movie == "No movies found":
        st.info("Please refine your filters or search to find movies.")
        return

    embeddings = generate_embeddings(movies_df['overview'])

    tab1, tab2 = st.tabs(["🎞️ Movie Info", "🎯 Recommendations"])

    selected_row = movies_df[movies_df['display'] == selected_movie].iloc[0]

    with tab1:
        left_col, right_col = st.columns([1, 2.5], gap="medium")
        with left_col:
            if selected_row['poster_path']:
                st.image(selected_row['poster_path'], width=220)
            else:
                st.info("No poster image available.")
        with right_col:
            st.markdown(f"### {selected_row['title']} ({selected_row['release_year']})")
            st.markdown(f"**Release Date:** {selected_row['release_date']}")
            st.markdown(f"**Rating:** ⭐ {selected_row['vote_average']}")
            st.markdown(f"**Overview:** {selected_row['overview']}")

        trailer_url = fetch_trailer(selected_row['id'])
        if trailer_url:
            st.video(trailer_url)
        else:
            st.info("No trailer available.")

    with tab2:
        if run_reco:
            recommendations = recommend(
                st.session_state['selected_movie'], movies_df, embeddings,
                top_n=top_n, min_rating=min_rating, min_year=min_year
            )
            st.session_state['recommendations'] = recommendations
        else:
            recommendations = st.session_state.get('recommendations', pd.DataFrame())

        if not recommendations.empty:
            st.markdown(f"### Recommendations based on: **{selected_movie}**")
            for _, row in recommendations.iterrows():
                with st.expander(f"{row['title']} ({row['release_year']})", expanded=False):
                    rec_cols = st.columns([1, 3], gap="small")
                    with rec_cols[0]:
                        if row['poster_path']:
                            st.image(row['poster_path'], width=120)
                        else:
                            st.info("No image available")
                    with rec_cols[1]:
                        st.markdown(f"**📅 Release Date:** {row['release_date']}")
                        st.markdown(f"**⭐ Rating:** {row['vote_average']}")
                        st.markdown(f"**📝 Overview:** {row['overview']}")
        else:
            st.info("Click 'Get Recommendations' to see movie suggestions!")

    st.markdown("""
        <hr style="border:1px solid #f0f0f0; margin-top: 3rem;">
        <div style='text-align: center; color: gray; font-size: 0.9rem;'>
            🚀 Developed by <b>Krishnaraj</b>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
