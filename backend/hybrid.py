import pandas as pd
from backend.content_based import content_recommend
from backend.collaborative import collaborative_recommend


def hybrid_recommend(movie_title, user_id, top_n=5):
    # Get recommendations
    content_df = content_recommend(movie_title, top_n)
    collab_df = collaborative_recommend(user_id, top_n)

    # Safety checks (NO ambiguous truth value)
    if content_df.empty and collab_df.empty:
        return pd.DataFrame(columns=["title"])

    if content_df.empty:
        return collab_df.head(top_n)

    if collab_df.empty:
        return content_df.head(top_n)

    # Combine results
    hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates()

    return hybrid_df.head(top_n)
