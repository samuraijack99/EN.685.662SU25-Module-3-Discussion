import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import joblib 

st.set_page_config(layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("netflix_dataset.csv")

df_netflix = load_data()

# Initialize session state for the page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "best_matches" not in st.session_state:
    st.session_state.best_matches = None

# Function to change the page
def change_page(page):
    st.session_state.current_page = page

# Sidebar for navigation
st.sidebar.title("Netflix Finder Navigation")
pages = ["Home", "Find by Filter", "About"]
for page in pages:
    if st.sidebar.button(page):
        change_page(page)

# Display content based on the selected page
if st.session_state.current_page == 'Home':

    st.markdown("""
        <div style="text-align: center;">
            <img src="https://images.ctfassets.net/y2ske730sjqp/5QQ9SVIdc1tmkqrtFnG9U1/de758bba0f65dcc1c6bc1f31f161003d/BrandAssets_Logos_02-NSymbol.jpg" width="300"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <div style="text-align: center;">
            <h1 style='color: red; font-size: 30px;'>Find your perfect Netflix content today</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    This app uses a Netflix dataset containing metadata of movies and TV shows that were released from 1925 to 2021. 
    It is designed to help the user search for content by title or filter content by release year to find information 
    about their desired movie(s) and tv show(s).
    """)
    
    # Direct search
    st.markdown("<span style='color: orange;'>Feel free to explore Netflix movies and tv shows by title, rating, release year, and more!</span>", unsafe_allow_html=True)

    # Direct search query
    query = st.text_input("If you already know what you're looking for, directly search by entering the title below:")

    # Press button to search by title
    if st.button("Search by Title"):

        # Function to fuzzy search a column in a df
        def fuzzy_search_column(query, df, column):
            column_vals = df[column].tolist()
            best_matches = process.extract(query, column_vals, limit=5)
            match_column_vals = [match[0] for match in best_matches]
            filtered_df = df[df[column].isin(match_column_vals)]

            return filtered_df, best_matches

        # Get best-matching titles
        search_filtered_df, best_matches = fuzzy_search_column(query, df_netflix, 'title')
        st.session_state.search_results = search_filtered_df
        st.session_state.best_matches = best_matches

    # If there are best-matching titles, display radio buttons to explore a title
    if not (st.session_state.search_results is None) and not (st.session_state.best_matches is None):
        st.write("Top matching titles:")
        title_options = {}
        
        for match, score in st.session_state.best_matches:
            title_options[f"{match} ({score}% match)"] = match

        display_options = st.radio(
            "Select to view full info:", 
            list(title_options.keys()), 
            key="search_radio")

        # Display information of selected title
        selected_title = title_options[display_options]
        selected_row = st.session_state.search_results[st.session_state.search_results['title'] == selected_title]
        if not selected_row.empty:
            row = selected_row.iloc[0]  # Get the first matching row
            #with st.expander("📄 Click to view full details"):
            for column, value in row.items():
                interesting_columns = ["title", "director", "cast", "rating", "duration", "description"]
                if column in interesting_columns:
                    st.markdown(f"**{column}:** {value}")
    else:
        st.write("No titles available.")

    # # Checkboxes to limit search by rating
    # st.markdown("<span style='color: orange;'>Select one or more ratings to include in your search.</span>", unsafe_allow_html=True)
    # ratings = df_netflix['rating'].unique()
    # checked_ratings = st.multiselect("Select one or more ratings:", ratings)

    # if checked_ratings:
    #     check_filtered_df = df_netflix[df_netflix['rating'].isin(checked_ratings)]
    # else:
    #     check_filtered_df = df_netflix.copy()

if st.session_state.current_page == 'Find by Filter':
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSeyLEBx_LwIxepNbfeNRO66tZPZWasbI8G_A&s" width="100"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="text-align: center;">
        <h1 style='color: red; font-size: 30px;'>Explore titles by filtering</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Create a slider filter by release year
    df_netflix_year = df_netflix.dropna(subset=['release_year']).copy()
    df_netflix_year['release_year'] = df_netflix_year['release_year'].astype(int)

    st.markdown(
        "<span style='color: orange;'>Use the slider below to select a range of release years and see how many movies/tv shows match:</span>",
        unsafe_allow_html=True
    )
    slider_years = st.slider(
        "Filter by release year:",
        min_value = df_netflix_year['release_year'].min(),
        max_value = df_netflix_year['release_year'].max(),
        value = (df_netflix_year['release_year'].min(), df_netflix_year['release_year'].max())
    )

    selected_df =  df_netflix[
                        (df_netflix['release_year'] >= slider_years[0]) & 
                        (df_netflix['release_year'] <= slider_years[1])
                    ]

    st.dataframe(selected_df, use_container_width=True)

    # Show graphic summary of content selected after filter
    total_content = len(df_netflix)
    selected_content = len(selected_df)
    st.markdown(f"Selected {selected_content} out of {total_content} movies/tv shows:")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [selected_content, total_content - selected_content], 
        colors=['darkRed', 'dimGray'], 
        startangle=90, 
        counterclock=False
    )

    ax.axis('equal')
    st.pyplot(fig, bbox_inches='tight', pad_inches=0)


if st.session_state.current_page == 'About':
        st.markdown("Built by Vian Ambar Agustono.")
        st.markdown("Dataset source:")
        st.markdown("https://www.kaggle.com/datasets/abhinavrongala/netflix-datasets-evaluation?select=Netflix+Datasets+Evaluation+MS+Excel.csv")
    