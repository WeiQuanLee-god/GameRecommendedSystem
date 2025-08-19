import streamlit as st
import pandas as pd
import numpy as np
import re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# SETTINGS
# =========================
CSV_PATH = "D:/AI_Project/games.csv"   

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
    return df

def build_content_matrix(df):
    text_corpus = df['category'] + ' ' + df['title'] + ' ' + df['description']
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    return vectorizer, tfidf_matrix

@st.cache_data(show_spinner=False)
def prepare_model(df: pd.DataFrame):
    vec, X = build_content_matrix(df)

    pop_norm = (np.log1p(df['installs']) - np.log1p(df['installs']).min()) / (
               np.log1p(df['installs']).max() - np.log1p(df['installs']).min() + 1e-9)
    rat_norm = (df['average_rating'] - df['average_rating'].min()) / (
               df['average_rating'].max() - df['average_rating'].min() + 1e-9)
    vol_norm = (np.log1p(df['total_ratings']) - np.log1p(df['total_ratings']).min()) / (
               np.log1p(df['total_ratings']).max() - np.log1p(df['total_ratings']).min() + 1e-9)

    boosts = pd.DataFrame({
        'popularity_boost': pop_norm.fillna(0),
        'rating_boost': rat_norm.fillna(0),
        'volume_boost': vol_norm.fillna(0)
    })
    return vec, X, boosts

def hybrid_score(base, boosts, alpha=0.7, beta=0.2, gamma=0.1):
    quality = 0.6 * boosts['rating_boost'].values + 0.4 * boosts['volume_boost'].values
    return alpha*base + beta*boosts['popularity_boost'].values + gamma*quality

def recommend(df, X, boosts, seed_indices, top_k=12, rule_filters=None, alpha=0.7, beta=0.2, gamma=0.1):
    base = np.zeros(len(df)) if len(seed_indices)==0 else cosine_similarity(X, X[seed_indices]).mean(axis=1).ravel()

    mask = np.ones(len(df), dtype=bool)
    if rule_filters:
        if rule_filters['price_type']=='Free':  mask &= (df['price']==0.0)
        elif rule_filters['price_type']=='Paid': mask &= (df['price']>0.0)
        if rule_filters['category']!='All':     mask &= (df['category']==rule_filters['category'])
        mask &= (df['average_rating']>=rule_filters['min_rating'])
        mask &= (df['installs']>=rule_filters['min_installs'])

    scores = hybrid_score(base, boosts, alpha, beta, gamma)
    scores[~mask] = -1
    if len(seed_indices): scores[seed_indices] = -1
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [i for i in top_idx if scores[i] >= 0], scores

# =========================
# UI
# =========================
st.set_page_config(page_title="Game Recommender System (Hybrid AI)", page_icon="👾", layout="wide")
st.title("🕹️ Top Games Recommender")
st.caption(f"Data source: `{CSV_PATH}` (auto‑reloads when the file changes)")

# Load data automatically (no button, no path input)
try:
    mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
    df = load_data(CSV_PATH, mtime)
    vec, X, boosts = prepare_model(df)
except Exception as e:
    st.error(f"Could not load dataset at {CSV_PATH}. Error: {e}")
    st.stop()


st.sidebar.header("🧷 Filters")
price_type = st.sidebar.selectbox("Price Type 💰", ["All","Free","Paid"])
categories = ["All"] + sorted(df['category'].unique().tolist())
chosen_cat = st.sidebar.selectbox("🎮 Category", categories)
min_rating = st.sidebar.slider("Minimum Rating ⭐", 0.0, 5.0, 3.5, 0.1)
min_installs = st.sidebar.number_input("Minimum Installs 📦", value=0, step=1000)

st.sidebar.header("⚖️ Weights")
alpha = st.sidebar.slider("Content Similarity (α)", 0.0, 1.0, 0.7, 0.05)
beta  = st.sidebar.slider("Popularity / Installs (β)", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("Quality: Rating + Volume (γ)", 0.0, 1.0, 0.1, 0.05)
total = alpha + beta + gamma or 1.0
alpha, beta, gamma = alpha/total, beta/total, gamma/total

rule_filters = dict(
    price_type=price_type,
    category=chosen_cat,
    min_rating=min_rating,
    min_installs=min_installs
)

# ---- Seed selection
st.subheader("1) Pick games you like 🎯")
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

# ---- Recommendations (auto updates when any control changes; no button needed)
st.subheader("2) Recommendations ✅")
k = st.number_input("Number of recommendations", min_value=5, max_value=40, value=12)

top_idx, scores = recommend(df, X, boosts, seed_idx, top_k=int(k), rule_filters=rule_filters,
                            alpha=alpha, beta=beta, gamma=gamma)

if not top_idx:
    st.warning("No matches found. Relax your filters.")
else:
    st.success(f"Found {len(top_idx)} recommendation(s).")
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
            st.markdown("**🧠 Why:** content similarity, popular downloads, high rating")
        st.markdown("---")


st.subheader("Or, browse trending 🔥")

hybrid_base = hybrid_score(np.zeros(len(df)), boosts, alpha=0, beta=0.7, gamma=0.3)

# Build mask as Series aligned to df.index
mask = pd.Series(True, index=df.index)
if price_type == "Free":
    mask &= (df['price'] == 0.0)
elif price_type == "Paid":
    mask &= (df['price'] > 0.0)
if chosen_cat != "All":
    mask &= (df['category'] == chosen_cat)
mask &= (df['average_rating'] >= min_rating)
mask &= (df['installs'] >= min_installs)

# Convert to positional boolean array
mask_arr = mask.to_numpy()

order = np.argsort(hybrid_base)[::-1]          # positions (0..N-1)
trending_idx = [i for i in order if mask_arr[i]][:10]

if trending_idx:
    cols = st.columns(2)
    for j, i in enumerate(trending_idx, start=1):
        row = df.iloc[i]                       # positional access
        with cols[(j-1) % 2]:
            st.markdown(
                f"**{j}. {row['title']}**  \n"
                f"⭐ {row['average_rating']:.2f} | "
                f"📦 {int(row['installs']):,}+ | "
                f"🎮 {row['category']}"
            )
else:
    st.info("No trending items match your filters.")
