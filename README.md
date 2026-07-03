ShopSense — Amazon-Style Recommendation Engine
A full-stack recommendation system trained on 7.8 million real Amazon electronics ratings, featuring three distinct recommendation approaches and an interactive web app.

Live demo: [coming soon]


What it does
ShopSense recommends products to Amazon shoppers using three layers:

Collaborative Filtering — ALS matrix factorization that learns 50 latent taste factors per user and product, purely from rating patterns
Content-Based Filtering — Cosine similarity engine that finds products similar to what a user already loved
Hybrid Model — Normalizes and combines both scores for stronger, more balanced recommendations
Cold Start — New users can rate products manually and get instant recommendations without being in the training data


Tech stack
Layer
Tools
Data processing
pandas, numpy
Collaborative filtering
implicit (ALS), scipy sparse matrices
Content-based filtering
scikit-learn (cosine similarity)
Web app
Streamlit
Language
Python 3.13



Dataset
Amazon Electronics ratings dataset — 7.8M ratings from 4.2M users across 476K products.

Applied 10-core filtering (iterative) to retain only users and products with ≥10 ratings, reducing to 347K high-quality interactions across 20K users and 11K products. This addresses the data sparsity problem common in recommendation systems.


How to run locally
1. Clone the repo

git clone https://github.com/Ocean1407/ShopSense.git

cd ShopSense

2. Create a virtual environment

python3 -m venv venv

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Download the dataset

Download ratings_Electronics.csv from Kaggle and place it in the data/ folder.

Then run the data cleaning notebook to generate ratings_clean.csv:

jupyter notebook notebooks/01_eda.ipynb

5. Run the app

streamlit run app.py


Project structure
shopsense/

├── data/

│   └── ratings_clean.csv       # Cleaned dataset (347K ratings)

├── notebooks/

│   └── 01_eda.ipynb            # EDA and data cleaning

├── src/

│   ├── collaborative.py        # ALS collaborative filtering

│   ├── content_based.py        # Cosine similarity content-based

│   └── hybrid.py               # Hybrid recommendation model

├── app.py                      # Streamlit web app

└── requirements.txt


Architecture
Raw Data (7.8M ratings)

        │

        ▼

10-Core Filtering

        │

        ▼

┌───────────────────────────────────┐

│         347K Clean Ratings        │

└───────────────────────────────────┘

        │                │

        ▼                ▼

Collaborative      Content-Based

 Filtering          Filtering

  (ALS)          (Cosine Similarity)

        │                │

        └────────┬───────┘

                 ▼

          Hybrid Model

       (Weighted Blend)

                 │

                 ▼

         Recommendations


Key concepts demonstrated
Matrix factorization — decomposing a sparse user-product matrix into latent factor embeddings
Cosine similarity — measuring product similarity in user-rating space
Hybrid recommendation — score normalization and weighted combination of multiple models
Cold start problem — content-based fallback for users not in the training data
Data sparsity — 10-core iterative filtering to ensure model quality


Results
Model
Approach
Use case
Collaborative
ALS, 50 factors, 20 iterations
Existing users with rating history
Content-Based
Cosine similarity on user-product matrix
Product-to-product similarity
Hybrid
50/50 normalized weighted blend
Best overall recommendations
Cold Start
Content-based on manual input
Brand new users



Future improvements
Fetch real product titles via Amazon Product API to replace illustrative labels
Add evaluation metrics (Precision@K, Recall@K, NDCG@K) comparison across all three models
Incorporate implicit feedback (clicks, views) in addition to explicit ratings
Experiment with neural collaborative filtering (NCF) as a fourth layer
Add user authentication so shoppers can save their rating history across sessions


