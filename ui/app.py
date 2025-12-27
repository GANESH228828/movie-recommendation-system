import sys
import os

# ---- Add project root to Python path ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

# ---- Backend imports ----
from backend.content_based import content_recommend
from backend.collaborative import collaborative_recommend
from backend.hybrid import hybrid_recommend

# ---- Load Data ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))

# ---- Streamlit UI ----
st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")

st.markdown("### Select Recommendation Type")
rec_type = st.radio(
    "Choose one:",
    ["Content-Based", "Collaborative", "Hybrid"]
)

movie_title = st.selectbox(
    "Select a movie",
    movies["title"].values
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=5
)

# ---- Recommendation Button ----
if st.button("Recommend"):
    st.subheader("📌 Recommended Movies")

    try:
        if rec_type == "Content-Based":
            recommendations = content_recommend(movie_title, top_n)

        elif rec_type == "Collaborative":
            recommendations = collaborative_recommend(user_id, top_n)

        else:
            recommendations = hybrid_recommend(user_id, movie_title, top_n)

        # ---- Proper DataFrame check ----
        if recommendations is None or recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.table(recommendations)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
