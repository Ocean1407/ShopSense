import pandas as pd
import numpy as np

def hybrid_recommendations(
    user_id,
    model,
    matrix,
    df_model,
    idx_to_product,
    item_similarity_df,
    df_clean,
    n=5,
    collab_weight=0.5,
    content_weight=0.5
):
    import sys
    sys.path.append('../src')
    from collaborative import get_recommendations as collab_recs
    from content_based import get_user_recommendations as content_recs

    # Get collaborative filtering recommendations
    collab_products, collab_scores = collab_recs(
        model, matrix, df_model, idx_to_product, user_id, n=20
    )

    # Get content-based recommendations
    content_products, content_scores = content_recs(
        item_similarity_df, df_clean, user_id, n=20
    )

    # Normalize scores to 0-1 range so they're comparable
    def normalize(scores):
        if len(scores) == 0:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    collab_scores_norm = normalize(collab_scores)
    content_scores_norm = normalize(content_scores)

    # Combine into one dictionary
    combined = {}
    for p, s in zip(collab_products, collab_scores_norm):
        combined[p] = combined.get(p, 0) + collab_weight * s

    for p, s in zip(content_products, content_scores_norm):
        combined[p] = combined.get(p, 0) + content_weight * s

    # Remove products user already rated
    already_rated = set(df_clean[df_clean['user_id'] == user_id]['product_id'].tolist())
    combined = {p: s for p, s in combined.items() if p not in already_rated}

    # Sort by combined score and return top n
    sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
    products = [r[0] for r in sorted_recs]
    scores = [r[1] for r in sorted_recs]

    return products, scores