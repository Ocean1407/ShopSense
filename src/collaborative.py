import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit import als

def load_data():
    df = pd.read_csv('../data/ratings_clean.csv')
    return df

def build_matrix(df):
    unique_users = sorted(df['user_id'].unique())
    unique_products = sorted(df['product_id'].unique())
    
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    product_to_idx = {p: i for i, p in enumerate(unique_products)}
    idx_to_product = {i: p for p, i in product_to_idx.items()}
    
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['product_idx'] = df['product_id'].map(product_to_idx)
    
    # rows = users, columns = products (this is what implicit expects)
    matrix = csr_matrix(
        (df['rating'], (df['user_idx'], df['product_idx'])),
        shape=(len(unique_users), len(unique_products))
    )
    
    return matrix, df, idx_to_product

def train_model(matrix):
    model = als.AlternatingLeastSquares(factors=50, iterations=20, random_state=42)
    model.fit(matrix)
    return model

def get_recommendations(model, matrix, df, idx_to_product, user_id, n=10):
    user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]
    
    user_items = matrix[user_idx]  # no more .T transpose needed
    
    valid_items = np.arange(matrix.shape[1])  # number of products is now columns
    
    ids, scores = model.recommend(
        user_idx,
        user_items,
        N=n,
        filter_already_liked_items=True,
        items=valid_items
    )
    
    recommended_products = [idx_to_product[int(i)] for i in ids]
    
    return recommended_products, scores