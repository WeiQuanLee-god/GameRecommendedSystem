# app.py
# Top Games Recommender – Content-Based + Collaborative Filtering + Hybrid
# OFFICIAL SUBMISSION (Single-member): Method = Content-Based | Algorithm = TF‑IDF + Cosine kNN
#
# Notes:
# - CF (Item‑Item co‑visitation kNN) and Hybrid are kept for demo/comparison,
#   but the *assessed* method/algorithm pair is Content‑Based using TF‑IDF + Cosine kNN.

import streamlit as st
import pandas as pd
import numpy as np
import os, re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# SETTINGS
CSV_PATH = "D:/AI_Project/games.csv"   # change if needed
SEED_DEFAULT_K = 12

# Helpers
def parse_installs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if '-' in s:
        s = s.split('-')[-1].strip()
    s = s.replace(',', '').replace('+', '').strip()
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*([KkMmBb])?$', s)
    if m:
        num = float(m.group(1)); suf = m.group(2)
        if not suf: return num
        return num * {'K':1e3,'k':1e3,'M':1e6,'m':1e6,'B':1e9,'b':1e9}[suf]
    digits = re.sub(r'\D', '', s)
    return float(digits) if digits else np.nan

@st.cache_data(show_spinner=False)
def load_data(csv_path: str, mtime: float):
    """Cache invalidates when file mtime changes."""
    df = pd.read_csv(csv_path)

    rename_map = {
        'average rating':'average_rating','Average Rating':'average_rating',
        'total ratings':'total_ratings','Total Ratings':'total_ratings',
        'Installs':'installs','installs+':'installs','installs':'installs',
        'Category':'category','Title':'title','Price':'price','Description':'description',
        'rank':'rank','Rank':'rank'
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})

    for col in ['title','category','price','installs','average_rating','total_ratings','rank']:
        if col not in df.columns: df[col] = np.nan
    if 'description' not in df.columns: df['description'] = ''

    df['installs'] = df['installs'].apply(parse_installs).fillna(0)
    for c in ['price','average_rating','total_ratings','rank']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['title'] = df['title'].astype(str)
    df['category'] = df['category'].astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    df.drop_duplicates(subset='title', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Content representation (ALGORITHM: TF‑IDF + Cosine kNN)
@st.cache_data(show_spinner=False)
def build_content_matrix(df):
    """
    Algorithm: TF‑IDF vectorization over title + category (+ optional description),
    with cosine similarity for k‑nearest‑neighbour retrieval.
    """
    text_corpus = df['category'] + ' ' + df['title'] + ' ' + df['description']
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    return vectorizer, tfidf_matrix

@st.cache_data(show_spinner=False)
def prepare_model(df: pd.DataFrame):
    vec, X = build_content_matrix(df)
    # Signals for Hybrid weighting (method, not algorithm)
    log_inst = np.log1p(df['installs'])
    pop_norm = (log_inst - log_inst.min()) / (log_inst.max() - log_inst.min() + 1e-9)
    rat_norm = (df['average_rating'] - df['average_rating'].min()) / (
               df['average_rating'].max() - df['average_rating'].min() + 1e-9)
    log_vol = np.log1p(df['total_ratings'])
    vol_norm = (log_vol - log_vol.min()) / (log_vol.max() - log_vol.min() + 1e-9)

    boosts = pd.DataFrame({
        'popularity_boost': pop_norm.fillna(0),
        'rating_boost': rat_norm.fillna(0),
        'volume_boost': vol_norm.fillna(0)
    })
    return vec, X, boosts

# Collaborative Filtering via Co‑Visitation (Item‑Item kNN)
@st.cache_data(show_spinner=False)
def build_item_item_cf(df: pd.DataFrame, window: int = 60) -> np.ndarray:
    """
    Method: Collaborative Filtering
    Algorithm variant: Item‑Item kNN using co‑visitation (synthetic sessions) with cosine normalization.
    """
    n = len(df)
    co = np.zeros((n, n), dtype=np.float32)

    def add_session(indices):
        L = len(indices)
        for i in range(L):
            a = indices[i]
            for j in range(i+1, L):
                b = indices[j]
                co[a, b] += 1.0
                co[b, a] += 1.0

    # Per‑category sessions
    cats = df['category'].fillna('UNKNOWN').unique().tolist()
    for c in cats:
        block = df.index[df['category'] == c].tolist()
        if not block: continue
        block_sorted = sorted(block, key=lambda i: (df.loc[i, 'rank'] if df.loc[i, 'rank'] > 0 else 10**9, -df.loc[i, 'installs']))
        for start in range(0, len(block_sorted), window):
            add_session(block_sorted[start:start+window])

    # Global sessions
    if 'rank' in df.columns and df['rank'].notna().any():
        ordered = df.index.tolist()
        ordered.sort(key=lambda i: (df.loc[i, 'rank'] if df.loc[i, 'rank'] > 0 else 10**9, -df.loc[i, 'installs']))
    else:
        ordered = df.sort_values('installs', ascending=False).index.tolist()
    for start in range(0, len(ordered), window):
        add_session(ordered[start:start+window])

    deg = co.sum(axis=1) + 1e-9
    norm = np.sqrt(np.outer(deg, deg))
    sim = co / norm
    np.fill_diagonal(sim, 1.0)
    return sim

def recommend_collab_item_item(df: pd.DataFrame, sim_mat: np.ndarray, seed_indices: List[int], top_k: int, mask: np.ndarray) -> Tuple[List[int], np.ndarray]:
    if len(seed_indices) == 0:
        degree = sim_mat.sum(axis=1)
        scores = degree.copy()
    else:
        scores = sim_mat[:, seed_indices].mean(axis=1)

    scores = np.asarray(scores).ravel().astype(float)
    if len(seed_indices):
        scores[seed_indices] = -1.0
    scores[~mask] = -1.0

    order = np.argsort(scores)[::-1]
    idx = [i for i in order if scores[i] >= 0][:top_k]
    return idx, scores

# Hybrid scoring (Method: Hybrid | Algorithm: Weighted Linear Fusion)
def hybrid_score(base, boosts, alpha=0.7, beta=0.2, gamma=0.1):
    quality = 0.6 * boosts['rating_boost'].values + 0.4 * boosts['volume_boost'].values
    return alpha*base + beta*boosts['popularity_boost'].values + gamma*quality

# Recommenders (Content, CF, Hybrid)
def recommend_content(df, X, seed_indices: List[int], top_k: int, mask: np.ndarray) -> Tuple[List[int], np.ndarray]:
    # Cosine kNN over TF‑IDF vectors (average if multiple seeds)
    base = np.zeros(len(df)) if len(seed_indices)==0 else cosine_similarity(X, X[seed_indices]).mean(axis=1).ravel()
    scores = base.copy()
    scores[~mask] = -1
    if len(seed_indices): scores[seed_indices] = -1
    order = np.argsort(scores)[::-1]
    idx = [i for i in order if scores[i] >= 0][:top_k]
    return idx, scores

def recommend_hybrid(df, X, boosts, seed_indices: List[int], top_k: int, mask: np.ndarray,
                     alpha=0.7, beta=0.2, gamma=0.1) -> Tuple[List[int], np.ndarray]:
    base = np.zeros(len(df)) if len(seed_indices)==0 else cosine_similarity(X, X[seed_indices]).mean(axis=1).ravel()
    scores = hybrid_score(base, boosts, alpha, beta, gamma)
    scores[~mask] = -1
    if len(seed_indices): scores[seed_indices] = -1
    order = np.argsort(scores)[::-1]
    idx = [i for i in order if scores[i] >= 0][:top_k]
    return idx, scores

# Evaluation helpers
def precision_recall_at_k(recommended: List[int], relevant: set, k: int) -> Tuple[float, float]:
    if k == 0:
        return 0.0, 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall

def diversity_gini(indices: List[int], X) -> float:
    if len(indices) < 2:
        return 0.0
    vecs = X[indices]
    sim = cosine_similarity(vecs)
    n = len(indices)
    upper = sim[np.triu_indices(n, 1)]
    avg_sim = upper.mean() if upper.size else 0.0
    return 1.0 - float(avg_sim)

# UI
st.set_page_config(page_title="Game Recommender (Content + CF + Hybrid)", page_icon="👾", layout="wide")
st.title("🕹️ Top Games Recommender – Content • Collaborative • Hybrid")

# Official method/algorithm banner (for marker clarity)
st.info("**Official Submission** — Method: Content-Based | Algorithm: TF‑IDF + Cosine kNN", icon="✅")

st.caption(f"Data source: `{CSV_PATH}` (auto‑reloads when the file changes)")

# Load data automatic
try:
    mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
    df = load_data(CSV_PATH, mtime)
    vec, X, boosts = prepare_model(df)
    sim_cf = build_item_item_cf(df, window=60)
except Exception as e:
    st.error(f"Could not load dataset at {CSV_PATH}. Error: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("🧷 Filters")
price_type = st.sidebar.selectbox("Price Type 💰", ["All","Free","Paid"])
categories = ["All"] + sorted(df['category'].unique().tolist())
chosen_cat = st.sidebar.selectbox("🎮 Category", categories)
min_rating = st.sidebar.slider("Minimum Rating ⭐", 0.0, 5.0, 3.5, 0.1)
min_installs = st.sidebar.number_input("Minimum Installs 📦", value=0, step=1000)

st.sidebar.header("⚙️ Method & Algorithm")
algo = st.sidebar.selectbox("Choose approach to **demo**", [
    "Content‑Based — TF‑IDF + Cosine kNN (Official)",
    "Collaborative Filtering — Item‑Item kNN (Co‑visitation + Cosine)",
    "Hybrid — Weighted Linear Fusion (Content + Popularity + Quality)"
])

st.sidebar.header("⚖️ Weights (Hybrid only)")
alpha = st.sidebar.slider("Content Similarity (α)", 0.0, 1.0, 0.7, 0.05)
beta  = st.sidebar.slider("Popularity / Installs (β)", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("Quality: Rating + Volume (γ)", 0.0, 1.0, 0.1, 0.05)
if "Hybrid" not in algo:
    alpha, beta, gamma = 0.7, 0.2, 0.1
else:
    total = alpha + beta + gamma or 1.0
    alpha, beta, gamma = alpha/total, beta/total, gamma/total

# Data selection
st.subheader("1) Pick games you like 🎯 (for personalized methods)")
seed_titles = st.multiselect("Choose games:", options=df['title'].tolist())
seed_idx = list(df.index[df['title'].isin(seed_titles)])

with st.expander("🔎 Search to find seeds"):
    kw = st.text_input("Keyword in title/category/description")
    if kw:
        hits = df[
            df['title'].str.contains(kw, case=False, na=False) |
            df['category'].str.contains(kw, case=False, na=False) |
            df['description'].str.contains(kw, case=False, na=False)
        ][['title','category','average_rating','installs']].head(30)
        st.dataframe(hits)

mask = pd.Series(True, index=df.index)
if price_type == "Free":
    mask &= (df['price'] == 0.0)
elif price_type == "Paid":
    mask &= (df['price'] > 0.0)
if chosen_cat != "All":
    mask &= (df['category'] == chosen_cat)
mask &= (df['average_rating'] >= min_rating)
mask &= (df['installs'] >= min_installs)
mask_arr = mask.to_numpy()

# Recommendations
st.subheader("2) Recommendations ✅")
k = st.number_input("Number of recommendations", min_value=5, max_value=40, value=SEED_DEFAULT_K)

if algo.startswith("Content‑Based"):
    top_idx, scores = recommend_content(df, X, seed_idx, int(k), mask_arr)
elif algo.startswith("Collaborative"):
    top_idx, scores = recommend_collab_item_item(df, sim_cf, seed_idx, int(k), mask_arr)
else:  # Hybrid
    top_idx, scores = recommend_hybrid(df, X, boosts, seed_idx, int(k), mask_arr, alpha, beta, gamma)

if not top_idx:
    st.warning("No matches found. Relax your filters or select some seeds (for personalized methods).")
else:
    st.success(f"Found {len(top_idx)} recommendation(s). Approach: {algo}")
    for rank, i in enumerate(top_idx, start=1):
        row = df.iloc[i]
        st.markdown(f"#### #{rank} – {row['title']}")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"**⭐ Rating:** {row['average_rating']:.2f}")
            st.markdown(f"**💬 Ratings:** {int(row['total_ratings']):,}")
        with cols[1]:
            st.markdown(f"**📦 Installs:** {int(row['installs']):,}+")
            price = row['price']; st.markdown(f"**💰 Price:** {'Free' if price==0 else f'USD {price:,.2f}'}")
        with cols[2]:
            st.markdown(f"**🎮 Category:** {row['category']}")
        with cols[3]:
            reason = {
                "Content‑Based — TF‑IDF + Cosine kNN (Official)": "content similarity over TF‑IDF vectors",
                "Collaborative Filtering — Item‑Item kNN (Co‑visitation + Cosine)": "items co‑appearing in sessions (cosine‑normalized)",
                "Hybrid — Weighted Linear Fusion (Content + Popularity + Quality)": "weighted blend of content similarity and popularity/quality signals"
            }[algo]
            st.markdown(f"**🧠 Why:** {reason}")
        st.markdown("---")

# Evaluation section
st.subheader("3) Evaluation 📊 (offline metrics)")
colA, colB, colC, colD = st.columns(4)

relevant = set()
if seed_idx:
    seed_cats = set(df.iloc[seed_idx]['category'].tolist())
    relevant = set(df.index[df['category'].isin(seed_cats)].tolist())

prec_k, rec_k = precision_recall_at_k(top_idx, relevant, int(k))
with colA:
    st.metric("Precision@k", f"{prec_k:.2f}")
with colB:
    st.metric("Recall@k", f"{rec_k:.2f}")
with colC:
    div = diversity_gini(top_idx, X)
    st.metric("Diversity (1-avgSim)", f"{div:.2f}")
with colD:
    coverage = len(set(top_idx)) / max(1, len(df))
    st.metric("Catalog Coverage", f"{coverage*100:.2f}%")

st.caption("Terminology: Methods = Content‑Based / Collaborative / Hybrid. Algorithms here are TF‑IDF + Cosine kNN (Content‑Based), Item‑Item kNN via co‑visitation + Cosine (Collaborative), and Weighted Linear Fusion (Hybrid).")
