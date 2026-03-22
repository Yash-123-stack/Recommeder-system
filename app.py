import streamlit as st
import pickle
import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
# On Streamlit Cloud, this pulls from "Settings > Secrets"
API_KEY = st.secrets.get("TMDB_API_KEY") if "TMDB_API_KEY" in st.secrets else os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="Hybrid Movie Matcher", 
    page_icon="🍿", 
    layout="wide"
)

# Function to load Lottie Animations with Error Handling
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load animations
lottie_popcorn = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_0yfs9t9p.json")

# Function to load external CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass 

# Apply Styling
local_css("assets/style.css")
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# --- 2. DATA UTILITIES ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

@st.cache_resource
def load_models():
    # We no longer load 'collaborative_brain.pkl' here to avoid surprise library dependency
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
    # Load the PRE-CALCULATED predictions dictionary
    user_preds = pickle.load(open('user_predictions.pkl', 'rb'))
    return movies_df, similarity_matrix, user_preds

# Handle potential file errors during load
try:
    movies, similarity, user_preds = load_models()
except FileNotFoundError as e:
    st.error(f"Missing data files: {e}. Ensure 'user_predictions.pkl' is uploaded to GitHub.")
    st.stop()

# --- 3. HYBRID LOGIC (LIGHTWEIGHT) ---
def get_hybrid_recommendations(movie_title):
    # Find index of selected movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Content Similarity Scores
    sim_scores = list(enumerate(similarity[idx]))
    
    # Hybrid calculation using pre-computed predictions
    hybrid_scores = []
    for i, score in sim_scores:
        m_id = movies.iloc[i].movieId
        # Grab the pre-calculated SVD prediction for User 1
        # Fallback to 3.0 if movie ID isn't in our dictionary
        prediction = user_preds.get(m_id, 3.0) 
        
        # Hybrid formula: Similarity weight + Personalization
        combined_score = (score * 5) + prediction
        hybrid_scores.append((i, combined_score))
    
    # Sort and take top 5 (skipping the input movie)
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recom_names = []
    recom_posters = []
    for i in hybrid_scores:
        movie_id = movies.iloc[i[0]].movieId
        recom_names.append(movies.iloc[i[0]].title)
        recom_posters.append(fetch_poster(movie_id))
    
    return recom_names, recom_posters

# --- 4. UI LAYOUT ---

# Hero Header
st.write('<div style="text-align: center; padding: 20px 0;">', unsafe_allow_html=True)
st.title("🎬 Yash Hybrid Movie Recommender")
st.write('<p style="font-size: 1.2rem; color: #9ca3af;">Optimized for Mind Spill | Content + Collaborative filtering</p>', unsafe_allow_html=True)
st.write('</div>', unsafe_allow_html=True)

# Search Area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_movie = st.selectbox(
        "Which movie do you love?", 
        movies['title'].values,
        index=None,
        placeholder="Search for a movie..."
    )
    search_button = st.button('Generate Recommendations', use_container_width=True)

st.markdown("---")

# --- 5. RESULTS & STATES ---

if not search_button and not selected_movie:
    col_anim1, col_anim2, col_anim3 = st.columns([1, 1, 1])
    with col_anim2:
        if lottie_popcorn:
            st_lottie(lottie_popcorn, height=250, key="initial")
        else:
            st.write("<h1 style='text-align:center;'>🍿</h1>", unsafe_allow_html=True)
        st.write("<p style='text-align: center; color: #6b7280;'>Ready to spill some movie magic?</p>", unsafe_allow_html=True)

elif search_button:
    if selected_movie:
        with st.status("🤖 Analyzing Hybrid Scores...", expanded=True) as status:
            st.write("Cross-referencing similarity data...")
            time.sleep(0.3)
            st.write("Mapping user preferences...")
            names, posters = get_hybrid_recommendations(selected_movie)
            status.update(label="Top 5 Matches Found!", state="complete", expanded=False)
        
        # DISPLAY GRID
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.image(posters[idx])
                st.write(f'<p class="stText" style="text-align: center; font-weight: bold; font-size: 0.9rem;">{names[idx]}</p>', unsafe_allow_html=True)
    else:
        st.warning("Please select a movie from the dropdown first!")

# Footer Branding
st.write('<p style="text-align: center; color: #4b5563; margin-top: 50px;">Built for <b>Mind Spill</b> | Powered by SVD & Cosine Similarity</p>', unsafe_allow_html=True)