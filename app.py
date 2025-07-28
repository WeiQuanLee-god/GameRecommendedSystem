import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Game Recommender System", page_icon="👾", layout="centered")

# Header
st.title("🕹️Game Recommender System")
st.markdown("Get personalized mobile game suggestions based on the top games from Google Play Store.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/AI_Project/games.csv")  # Make sure this path is correct
    df.drop_duplicates(subset="title", inplace=True)
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options 🧷")

# Free vs Paid
price_filter = st.sidebar.selectbox(
    " Price Type💰:",
    ("All", "Free", "Paid")
)

if price_filter == "Free":
    filtered_df = df[df['price'] == 0.0]
elif price_filter == "Paid":
    filtered_df = df[df['price'] > 0.0]
else:
    filtered_df = df.copy()


# Category Filter
all_genres = sorted(df['category'].unique())
all_genres.insert(0, "All")

selected_genre = st.sidebar.selectbox("🎮 Select Game Category:", all_genres)

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['category'] == selected_genre]

# Search Bar
st.subheader("Search Games🔎")
search_input = st.text_input("Enter game name or keyword:")

if search_input:
    search_df = filtered_df[filtered_df['title'].str.contains(search_input, case=False)]
else:
    search_df = filtered_df

# Display Section
if not search_df.empty:
    st.success(f"Found {len(search_df)} result(s).")
    for _, row in search_df.iterrows():
        with st.container():
            st.markdown(f"###  {row['title']}")
            st.markdown(f"- 🌟 **Rating:** {row['average rating']}")
            st.markdown(f"- 📦 **Installs:** {row['installs']}+")
            st.markdown(f"- 🧩 **Category:** {row['category']}")
            st.markdown(f"- 💬 **Total Ratings:** {row['total ratings']:,}")
            st.markdown(f"- 💰 **Price:** {'Free' if row['price'] == 0 else f'USD {row['price']}'}")
            st.markdown("---")
else:
    st.warning("No matching games found. Try a different keyword or adjust the filters.")
