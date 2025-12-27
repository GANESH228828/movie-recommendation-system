import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("data/movies.csv")

# Preprocess genres
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def content_recommend(movie_title, top_n=5):
    if movie_title not in indices:
        return pd.DataFrame(columns=["title"])

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][["title"]]
