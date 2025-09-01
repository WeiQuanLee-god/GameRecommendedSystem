import streamlit as st
import pandas as pd
import numpy as np
import re, os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack as sp_vstack

# =========================
# SETTINGS
# =========================
CSV_PATH = "D:/AI_Project/games.csv"   # change if needed
SEED_DEFAULT_K = 12

# =========================
# Helpers
# =========================
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
        'Installs':'installs','installs+':'installs',
        'Category':'category','Title':'title','Price':'price','Description':'description'
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})

    for col in ['title','category','price','installs','average_rating','total_ratings']:
        if col not in df.columns: df[col] = np.nan
    if 'description' not in df.columns: df['description'] = ''

    df['installs'] = df['installs'].apply(parse_installs).fillna(0)
    for c in ['price','average_rating','total_ratings']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['title'] = df['title'].astype(str)
    df['category'] = df['category'].astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    df.drop_duplicates(subset='title', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ===== Content representation
@st.cache_data(show_spinner=False)
def build_content_matrix(df):
    text_corpus = df['category'] + ' ' + df['title'] + ' ' + df['description']
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    return vectorizer, tfidf_matrix

@st.cache_data(show_spinner=False)
def prepare_model(df: pd.DataFrame):
    vec, X = build_content_matrix(df)
    # popularity & quality features
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

# ===== Scoring utilities

def hybrid_score(base, boosts, alpha=0.7, beta=0.2, gamma=0.1):
    quality = 0.6 * boosts['rating_boost'].values + 0.4 * boosts['volume_boost'].values
    return alpha*base + beta*boosts['popularity_boost'].values + gamma*quality

# ===== Recommenders (4 approaches incl. SGD)

def recommend_content(df, X, seed_indices: List[int], top_k: int, mask: np.ndarray) -> Tuple[List[int], np.ndarray]:
    base = np.zeros(len(df)) if len(seed_indices)==0 else cosine_similarity(X, X[seed_indices]).mean(axis=1).ravel()
    scores = base.copy()
    scores[~mask] = -1
    if len(seed_indices): scores[seed_indices] = -1
    order = np.argsort(scores)[::-1]
    idx = [i for i in order if scores[i] >= 0][:top_k]
    return idx, scores


def recommend_popularity(df, boosts, top_k: int, mask: np.ndarray) -> Tuple[List[int], np.ndarray]:
    # non‑personalized: popularity + quality
    scores = 0.7*boosts['popularity_boost'].values + 0.3*(0.6*boosts['rating_boost'].values + 0.4*boosts['volume_boost'].values)
    scores = scores.astype(float)
    scores[~mask] = -1
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


def recommend_sgd_linear(df, X, seed_indices: List[int], top_k: int, mask: np.ndarray,
                          n_neg: int = 500, alpha: float = 1e-4, loss: str = "log_loss") -> Tuple[List[int], np.ndarray]:
    """
    Train a quick personalized linear model with SGD.
    Positives = user's selected seeds; Negatives = random non-seed items.
    Scores are decision_function over all items.
    """
    n = len(df)
    if len(seed_indices) == 0:
        return [], np.full(n, -1.0)
    rng = np.random.default_rng(42)
    all_idx = np.arange(n)
    neg_pool = np.setdiff1d(all_idx, np.array(seed_indices), assume_unique=False)
    n_neg = int(min(n_neg, len(neg_pool)))
    if n_neg <= 0:
        return [], np.full(n, -1.0)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)

    X_train = sp_vstack([X[seed_indices], X[neg_idx]])
    y = np.concatenate([np.ones(len(seed_indices)), np.zeros(n_neg)])

    clf = SGDClassifier(loss=loss, alpha=alpha, max_iter=1000, tol=1e-3, class_weight="balanced", random_state=42)
    clf.fit(X_train, y)

    # Decision scores for all items
    try:
        scores = clf.decision_function(X)
    except Exception:
        # Some losses expose predict_proba only; fallback to that
        proba = clf.predict_proba(X)[:, 1]
        scores = proba

    scores = np.asarray(scores, dtype=float)
    scores[~mask] = -1
    if len(seed_indices):
        scores[seed_indices] = -1
    order = np.argsort(scores)[::-1]
    idx = [i for i in order if scores[i] >= 0][:top_k]
    return idx, scores

# ===== Evaluation helpers

def precision_recall_at_k(recommended: List[int], relevant: set, k: int) -> Tuple[float, float]:
    if k == 0:
        return 0.0, 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def diversity_gini(indices: List[int], X) -> float:
    # higher means more diverse
    if len(indices) < 2:
        return 0.0
    vecs = X[indices]
    sim = cosine_similarity(vecs)
    n = len(indices)
    # exclude diagonal
    upper = sim[np.triu_indices(n, 1)]
    avg_sim = upper.mean() if upper.size else 0.0
    return 1.0 - float(avg_sim)

# =========================
# UI
# =========================
st.set_page_config(page_title="Game Recommender (Multi‑approach)", page_icon="👾", layout="wide")
st.title("🕹️ Top Games Recommender – Multi‑Approach + Evaluation")
st.caption(f"Data source: `{CSV_PATH}` (auto‑reloads when the file changes)")

# Load data automatic
try:
    mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
    df = load_data(CSV_PATH, mtime)
    vec, X, boosts = prepare_model(df)
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

st.sidebar.header("⚙️ Algorithm")
algo = st.sidebar.selectbox("Choose approach", [
    "Content‑Based (TF‑IDF)",
    "Popularity + Quality (Non‑personalized)",
    "Hybrid (Content + Popularity + Quality)",
    "Personalized SGD (linear)"
])

# SGD algorithms
if algo == "Personalized SGD (linear)":
    sgd_neg = st.sidebar.slider("SGD negative samples", 100, 2000, 500, 50)
    sgd_alpha = st.sidebar.select_slider("SGD regularization α", options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-4)
    sgd_loss = st.sidebar.selectbox("SGD loss", ["log_loss", "hinge"])  # logistic vs linear SVM
else:
    sgd_neg, sgd_alpha, sgd_loss = 500, 1e-4, "log_loss"

st.sidebar.header("⚖️ Weights (Hybrid only)")
alpha = st.sidebar.slider("Content Similarity (α)", 0.0, 1.0, 0.7, 0.05)
beta  = st.sidebar.slider("Popularity / Installs (β)", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("Quality: Rating + Volume (γ)", 0.0, 1.0, 0.1, 0.05)
if algo != "Hybrid (Content + Popularity + Quality)":
    alpha, beta, gamma = 0.7, 0.2, 0.1
else:
    total = alpha + beta + gamma or 1.0
    alpha, beta, gamma = alpha/total, beta/total, gamma/total

rule_filters = dict(
    price_type=price_type,
    category=chosen_cat,
    min_rating=min_rating,
    min_installs=min_installs
)

# Data selection
st.subheader("1) Pick games you like 🎯 (for content/hybrid)")
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

if algo == "Content‑Based (TF‑IDF)":
    top_idx, scores = recommend_content(df, X, seed_idx, int(k), mask_arr)
elif algo == "Popularity + Quality (Non‑personalized)":
    top_idx, scores = recommend_popularity(df, boosts, int(k), mask_arr)
elif algo == "Hybrid (Content + Popularity + Quality)":
    top_idx, scores = recommend_hybrid(df, X, boosts, seed_idx, int(k), mask_arr, alpha, beta, gamma)
else:
    top_idx, scores = recommend_sgd_linear(df, X, seed_idx, int(k), mask_arr, n_neg=sgd_neg, alpha=float(sgd_alpha), loss=sgd_loss)

if not top_idx:
    st.warning("No matches found. Relax your filters or select some seeds (for content/hybrid).")
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
                "Content‑Based (TF‑IDF)": "content similarity (title/desc/category)",
                "Popularity + Quality (Non‑personalized)": "top installs + high rating & votes",
                "Hybrid (Content + Popularity + Quality)": "content similarity + installs + rating"
            }[algo]
            st.markdown(f"**🧠 Why:** {reason}")
        st.markdown("---")

# Evaluation section
st.subheader("3) Evaluation 📊 (offline metrics)")
colA, colB, colC, colD = st.columns(4)

# Define pseudo‑relevance - items in the same category as any seed are considered relevant
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

st.caption("Notes: Precision/Recall computed vs. items sharing the same category as selected seeds (proxy ground truth). RMSE is not reported because the dataset lacks user‑specific ratings. If you add a user‑item rating matrix later, use RMSE/MSE for predictions.")

# Simple user feedback collection
st.subheader("4) Quick User Feedback ✍️")
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = []

fb_col1, fb_col2 = st.columns([3,1])
with fb_col1:
    fb_text = st.text_input("Tell us if these recommendations are useful (optional)")
with fb_col2:
    if st.button("Submit Feedback"):
        if fb_text.strip():
            st.session_state['feedback'].append(fb_text.strip())
            st.success("Thanks! Your feedback has been recorded locally for this demo.")
        else:
            st.info("Please enter some feedback text before submitting.")

if st.session_state['feedback']:
    st.write("**Collected Feedback (session):**")
    for i, t in enumerate(st.session_state['feedback'], 1):
        st.write(f"{i}. {t}")

# Trending browser (uses popularity/quality)
st.subheader("Or, browse trending 🔥")

pq_scores = 0.7*boosts['popularity_boost'].values + 0.3*(0.6*boosts['rating_boost'].values + 0.4*boosts['volume_boost'].values)
order = np.argsort(pq_scores)[::-1]
trending_idx = [i for i in order if mask_arr[i]][:10]

if trending_idx:
    cols = st.columns(2)
    for j, i in enumerate(trending_idx, start=1):
        row = df.iloc[i]
        with cols[(j-1) % 2]:
            st.markdown(
                f"**{j}. {row['title']}**  \n"
                f"⭐ {row['average_rating']:.2f} | "
                f"📦 {int(row['installs']):,}+ | "
                f"🎮 {row['category']}"
            )
else:
    st.info("No trending items match your filters.")
