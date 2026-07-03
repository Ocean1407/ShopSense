import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_product_name(asin):
    categories = [
        "Wireless Headphones", "Bluetooth Speaker", "USB Cable", "Phone Case",
        "Screen Protector", "Laptop Stand", "Mechanical Keyboard", "Gaming Mouse",
        "Webcam", "Monitor", "External SSD", "Memory Card", "Smart Watch",
        "Earbuds", "Charging Pad", "HDMI Cable", "USB Hub", "Desk Lamp",
        "Portable Charger", "Tablet Stand", "Camera Lens", "Action Camera",
        "Smart Plug", "LED Strip", "Noise Cancelling Headset", "Graphics Card",
        "RAM Module", "CPU Cooler", "Cable Management Kit", "Router"
    ]
    brands = [
        "Anker", "Logitech", "Sony", "Samsung", "Bose", "JBL", "Belkin",
        "SanDisk", "Seagate", "Corsair", "Razer", "HyperX", "Jabra",
        "Plantronics", "AmazonBasics", "TP-Link", "Netgear", "ASUS", "LG", "Dell"
    ]
    idx1 = sum(ord(c) for c in asin) % len(brands)
    idx2 = sum(ord(c) for c in asin[::-1]) % len(categories)
    return f"{brands[idx1]} {categories[idx2]}"

def build_item_similarity(df_clean):
    # Use only top 2000 most active users to stay within cloud memory limits
    top_users = df_clean['user_id'].value_counts().head(2000).index
    df_sample = df_clean[df_clean['user_id'].isin(top_users)]
    
    item_user_matrix = df_sample.pivot_table(
        index='product_id',
        columns='user_id',
        values='rating'
    ).fillna(0)
    
    # Only keep products that have at least 1 rating from these users
    item_user_matrix = item_user_matrix[item_user_matrix.sum(axis=1) > 0]
    
    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )
    return item_similarity_df

def get_similar_products(item_similarity_df, product_id, n=5):
    if product_id not in item_similarity_df.index:
        return [], []
    similar = item_similarity_df[product_id].sort_values(ascending=False)[1:n+1]
    return similar.index.tolist(), similar.values.tolist()

def get_user_recommendations(item_similarity_df, df_clean, user_id, n=5):
    user_rated = df_clean[df_clean['user_id'] == user_id]
    top_rated = user_rated.sort_values('rating', ascending=False)['product_id'].head(3).tolist()
    all_recommendations = {}
    for product_id in top_rated:
        similar_products, scores = get_similar_products(item_similarity_df, product_id, n=10)
        for p, s in zip(similar_products, scores):
            if p not in user_rated['product_id'].values:
                all_recommendations[p] = max(all_recommendations.get(p, 0), s)
    sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    products = [r[0] for r in sorted_recs]
    scores = [r[1] for r in sorted_recs]
    return products, scores

def get_coldstart_recommendations(item_similarity_df, user_ratings, n=5):
    all_recommendations = {}
    sorted_ratings = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
    top_rated = [p for p, r in sorted_ratings if r >= 3][:5]
    for product_id in top_rated:
        if product_id not in item_similarity_df.index:
            continue
        similar_products, scores = get_similar_products(item_similarity_df, product_id, n=15)
        for p, s in zip(similar_products, scores):
            if p not in user_ratings:
                weight = user_ratings[product_id] / 5.0
                all_recommendations[p] = max(
                    all_recommendations.get(p, 0),
                    s * weight
                )
    sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    products = [r[0] for r in sorted_recs]
    scores = [r[1] for r in sorted_recs]
    return products, scores