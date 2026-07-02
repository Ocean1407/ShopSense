import streamlit as st
import pandas as pd
import sys
sys.path.append('src')

from collaborative import load_data, build_matrix, train_model, get_recommendations
from content_based import build_item_similarity, get_user_recommendations, get_coldstart_recommendations, generate_product_name
from hybrid import hybrid_recommendations

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="ShopSense", page_icon=None, layout="wide")

# ── Custom styling ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 20% 0%, #2a1d5e 0%, #120c2e 45%, #0a0718 100%);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1340 0%, #120c2e 100%);
        border-right: 1px solid #3a2d7a;
    }
    section[data-testid="stSidebar"] * {
        color: #e8e6f7 !important;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 18px rgba(190,120,255,0.25); }
        50% { box-shadow: 0 0 28px rgba(190,120,255,0.45); }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1rem 1rem 3rem 1rem;
        animation: fadeInUp 0.6s ease;
    }
    .hero-title {
        font-size: 4.4rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #ff6ec4, #b06aff, #5ad1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.05;
    }
    .hero-subtitle {
        color: #cfc9ec;
        font-size: 1.25rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #c08aff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f5f3ff;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .section-desc {
        color: #b6b0d6;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-bottom: 1.4rem;
    }

    .approach-card {
        background: linear-gradient(160deg, #1d1648 0%, #161038 100%);
        border: 1px solid #3a2d7a;
        border-radius: 18px;
        padding: 1.6rem;
        height: 100%;
        animation: fadeInUp 0.6s ease;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .approach-card:hover {
        transform: translateY(-4px);
        border-color: #b06aff;
        box-shadow: 0 0 24px rgba(176,106,255,0.3);
    }
    .approach-tag {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.35rem 0.8rem;
        border-radius: 100px;
        margin-bottom: 0.9rem;
    }
    .tag-collab { background-color: rgba(90,209,255,0.15); color: #5ad1ff; }
    .tag-content { background-color: rgba(255,110,196,0.15); color: #ff6ec4; }
    .tag-hybrid { background-color: rgba(176,106,255,0.18); color: #c08aff; }

    .approach-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f5f3ff;
        margin-bottom: 0.45rem;
    }
    .approach-desc {
        color: #b6b0d6;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-block;
        background: rgba(176,106,255,0.15);
        border: 1px solid #b06aff;
        color: #d4b6ff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.55rem 1.2rem;
        border-radius: 100px;
        margin: 1.5rem 0;
        animation: fadeInUp 0.5s ease, glow 2.5s infinite ease-in-out;
    }

    .info-card {
        background: linear-gradient(160deg, #1d1648 0%, #161038 100%);
        border: 1px solid #3a2d7a;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.6rem;
        animation: fadeInUp 0.5s ease;
    }

    .rec-card {
        background: linear-gradient(160deg, #1d1648 0%, #161038 100%);
        border: 1px solid #3a2d7a;
        border-radius: 16px;
        padding: 1.3rem 1.7rem;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        animation: fadeInUp 0.4s ease;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateX(6px);
        border-color: #b06aff;
        box-shadow: 0 0 22px rgba(176,106,255,0.3);
    }
    .rec-rank {
        font-size: 1.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6ec4, #b06aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        min-width: 58px;
    }
    .rec-id {
        font-family: 'SF Mono', Menlo, monospace;
        font-size: 0.95rem;
        color: #f5f3ff;
        font-weight: 700;
    }
    .rec-name {
        font-size: 0.88rem;
        color: #b6b0d6;
        margin-top: 0.15rem;
    }
    .rec-score {
        color: #6e5fa0;
        font-size: 0.82rem;
    }
    .rec-link {
        color: #5ad1ff;
        font-size: 0.85rem;
        text-decoration: none;
        font-weight: 600;
        white-space: nowrap;
    }
    .rec-link:hover { text-decoration: underline; }

    .sidebar-glow-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6ec4, #b06aff, #5ad1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.6rem;
    }
    .mode-card {
        background: rgba(176,106,255,0.1);
        border: 1px solid #3a2d7a;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        font-size: 0.85rem;
        color: #cfc9ec !important;
        line-height: 1.5;
        margin-top: 1rem;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #ff6ec4, #b06aff);
        color: white;
        border-radius: 100px;
        border: none;
        padding: 0.75rem 2.1rem;
        font-weight: 700;
        font-size: 0.97rem;
        transition: box-shadow 0.25s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 24px rgba(176,106,255,0.5);
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1d1648;
        border-color: #3a2d7a;
        color: #f5f3ff;
    }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stCaption, p, span, label { color: #cfc9ec; }
    h1, h2, h3 { color: #f5f3ff; }
    hr { border-color: #3a2d7a; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">ShopSense</div>
    <div class="hero-subtitle">A recommendation engine trained on 7.8 million real Amazon ratings</div>
</div>
""", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────
st.markdown('<p class="section-label">Overview</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">How this works</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">This app recommends products to real, anonymized Amazon shoppers using three distinct approaches.</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="approach-card">
        <span class="approach-tag tag-collab">Collaborative</span>
        <div class="approach-name">Similar shoppers</div>
        <div class="approach-desc">People with similar taste to you bought this.</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="approach-card">
        <span class="approach-tag tag-content">Content-Based</span>
        <div class="approach-name">Similar products</div>
        <div class="approach-desc">This is similar to things you've already liked.</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="approach-card">
        <span class="approach-tag tag-hybrid">Hybrid</span>
        <div class="approach-name">Best of both</div>
        <div class="approach-desc">Combines both approaches for a stronger result.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
    <p class="section-desc" style="margin-bottom:0;">
        <b style="color:#f5f3ff;">To get started:</b> pick a shopper below, choose a recommendation 
        mode in the sidebar, then generate recommendations. Try switching modes for the same user 
        to see how each approach differs.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load and cache everything ─────────────────────────────
@st.cache_resource
def setup():
    df_clean = pd.read_csv('data/ratings_clean.csv')
    matrix, df_model, idx_to_product = build_matrix(df_clean.copy())
    model = train_model(matrix)
    item_similarity_df = build_item_similarity(df_clean)
    return df_clean, matrix, df_model, idx_to_product, model, item_similarity_df

with st.spinner("Loading models — this takes about 30 seconds on first run"):
    df_clean, matrix, df_model, idx_to_product, model, item_similarity_df = setup()

st.markdown('<span class="status-pill">Models ready</span>', unsafe_allow_html=True)

# Define early so available throughout
all_products = sorted(df_clean['product_id'].unique().tolist())
product_name_lookup = {p: f"{generate_product_name(p)} ({p})" for p in all_products}
asin_from_display = {v: k for k, v in product_name_lookup.items()}

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown('<p class="sidebar-glow-title">Settings</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="section-label" style="color:#c08aff;">Recommendation mode</p>', unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Mode",
    ["Hybrid", "Collaborative", "Content-Based"],
    label_visibility="collapsed"
)

mode_explainer = {
    "Hybrid": "Combines collaborative and content-based scores for a stronger, balanced result.",
    "Collaborative": "Finds shoppers with similar taste to this person and recommends what they liked.",
    "Content-Based": "Looks at this shopper's favourite products and finds similar items."
}
st.sidebar.markdown(f'<div class="mode-card">{mode_explainer[mode]}</div>', unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown('<p class="section-label" style="color:#c08aff;">Number of recommendations</p>', unsafe_allow_html=True)
n_recs = st.sidebar.slider("Recs", 3, 10, 5, label_visibility="collapsed")

# ── User selection ────────────────────────────────────────
st.markdown('<p class="section-label">Step 1</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Choose a shopper</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">These are real, anonymized Amazon shoppers from the dataset, sorted by activity.</p>', unsafe_allow_html=True)

sample_users = df_clean['user_id'].value_counts().head(50).index.tolist()
selected_user = st.selectbox("Choose a user ID:", sample_users, label_visibility="collapsed")

user_history = df_clean[df_clean['user_id'] == selected_user][['product_id', 'rating']]
st.caption(f"This shopper has rated {len(user_history)} products in our dataset.")
st.dataframe(user_history.reset_index(drop=True), height=180, use_container_width=True)

# ── Generate recommendations ──────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 2</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Generate recommendations</p>', unsafe_allow_html=True)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        if mode == "Hybrid":
            products, scores = hybrid_recommendations(
                selected_user, model, matrix, df_model,
                idx_to_product, item_similarity_df, df_clean, n=n_recs
            )
        elif mode == "Collaborative":
            products, scores = get_recommendations(
                model, matrix, df_model, idx_to_product, selected_user, n=n_recs
            )
        else:
            products, scores = get_user_recommendations(
                item_similarity_df, df_clean, selected_user, n=n_recs
            )

    st.markdown(f"<br><p class='section-title' style='font-size:1.3rem;'>Top {n_recs} recommendations — {mode}</p>", unsafe_allow_html=True)

    for i, (p, s) in enumerate(zip(products, scores), 1):
        name = generate_product_name(p)
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-rank">{i:02d}</div>
            <div style="flex:1;">
                <div class="rec-id">{name}</div>
                <div class="rec-name">{p}</div>
                <div class="rec-score">Score {s:.3f}</div>
            </div>
            <a class="rec-link" href="https://www.amazon.com/s?k={name.replace(' ', '+')}" target="_blank">Search on Amazon →</a>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Scores reflect relative ranking confidence between products, not an absolute percentage.")
else:
    st.caption("Click the button above once you've picked a shopper to see recommendations.")

# ── Cold Start — New User Input ───────────────────────────
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Try it yourself</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Get your own recommendations</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">Not in our dataset? No problem. Rate a few products below and we\'ll recommend something just for you — no account needed.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card" style="margin-bottom:1.2rem;">
    <p class="section-desc" style="margin-bottom:0; font-size:0.88rem;">
        <b style="color:#f5f3ff;">Note:</b> Product names are illustrative labels assigned for demo purposes.
        Recommendations are based on real Amazon rating patterns from 7.8M ratings.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-desc">Pick up to 5 products and rate them 1–5 stars:</p>', unsafe_allow_html=True)

user_ratings = {}
for i in range(1, 6):
    col_a, col_b = st.columns([0.7, 0.3])
    with col_a:
        display = st.selectbox(
            f"Product {i}",
            options=["— skip —"] + list(product_name_lookup.values()),
            key=f"product_{i}",
            label_visibility="collapsed"
        )
    with col_b:
        rating = st.slider(
            f"Rating {i}",
            1, 5, 3,
            key=f"rating_{i}",
            label_visibility="collapsed"
        )
    if display != "— skip —":
        actual_asin = asin_from_display[display]
        user_ratings[actual_asin] = rating

if st.button("Get My Recommendations", key="coldstart_btn"):
    if len(user_ratings) == 0:
        st.warning("Please select at least one product above.")
    else:
        with st.spinner("Finding recommendations based on your ratings..."):
            cold_products, cold_scores = get_coldstart_recommendations(
                item_similarity_df, user_ratings, n=n_recs
            )

        if len(cold_products) == 0:
            st.warning("Not enough data to generate recommendations. Try rating more products.")
        else:
            st.markdown(f"<br><p class='section-title' style='font-size:1.3rem;'>Your top {len(cold_products)} picks</p>", unsafe_allow_html=True)

            for i, (p, s) in enumerate(zip(cold_products, cold_scores), 1):
                name = generate_product_name(p)
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-rank">{i:02d}</div>
                    <div style="flex:1;">
                        <div class="rec-id">{name}</div>
                        <div class="rec-name">{p}</div>
                        <div class="rec-score">Score {s:.3f}</div>
                    </div>
                    <a class="rec-link" href="https://www.amazon.com/s?k={name.replace(' ', '+')}" target="_blank">Search on Amazon →</a>
                </div>
                """, unsafe_allow_html=True)

            st.caption("Recommendations are based on products similar to your highest-rated picks.")