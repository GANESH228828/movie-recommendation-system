import pandas as pd

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")


def collaborative_recommend(user_id, top_n=5):
    if user_id not in ratings["userId"].values:
        return pd.DataFrame(columns=["title"])

    # Movies already rated by user
    user_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    # Other users who watched same movies
    similar_users = ratings[ratings["movieId"].isin(user_movies)]["userId"].unique()

    # Movies liked by similar users
    recommendations = ratings[
        (ratings["userId"].isin(similar_users)) &
        (~ratings["movieId"].isin(user_movies))
    ]

    top_movies = (
        recommendations.groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return movies[movies["movieId"].isin(top_movies)][["title"]]
