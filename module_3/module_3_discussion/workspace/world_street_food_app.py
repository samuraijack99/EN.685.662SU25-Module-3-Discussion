import streamlit as st
import pandas as pd
import time  # Importing time for simulating loading
import requests
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib 

st.set_page_config(layout="wide")

# Initialize session state for the page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Function to change the page
def change_page(page):
    st.session_state.current_page = page

# Sidebar for navigation
st.sidebar.title("Spotify App Navigation")
pages = ["Home", "Data Viewer", "Explore Songs", "Find a Song", "Popularity Model", "About Us"]
for page in pages:
    if st.sidebar.button(page):
        change_page(page)

# Display content based on the selected page
if st.session_state.current_page == 'Home':

    st.markdown("<h1 style='color: red; font-size: 65px;'>Welcome to the Spotify Song Explorer</h1>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .custom-font {
        font-family: 'Courier New', Courier, monospace;
        font-size: 24px;
        color: #FF5733;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="custom-font">This app serves as an educational tool to demonstrate UX/UI design principles and data visualization techniques. It showcases how data can be organized and filtered to offer a smooth user experience and enable insightful analysis.</p>', unsafe_allow_html=True)

    st.divider()
    
    st.markdown("""
    <style>
    .custom-font-2 {
        font-family: Verdana, sans-serif;
        font-size: 16px;
        color: white; 
    }  
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="custom-font-2">In this exploration, we’re going to apply usability heuristics to showcase how thoughtful design choices can make key elements stand out effectively. By focusing on principles like visibility, feedback, and aesthetic minimalist design, we aim to enhance the user’s experience, ensuring that important information captures attention while maintaining a clean, intuitive layout. Through Streamlit, we’ll see how these design choices come together to create an engaging and accessible interface.</p>', unsafe_allow_html=True)
    # Changing text color using markdown
    st.markdown("<h1 style='color: blue;'>Dynamic Header with Color</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: green;'>Subheader with Different Color</h2>", unsafe_allow_html=True)

    # Experimenting with different sizes
    st.markdown("<h1 style='font-size: 50px;'>Large Header</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 35px;'>Medium Header</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 25px;'>Small Header</h3>", unsafe_allow_html=True)

    # Using different fonts (Basic with custom CSS styling)
    st.markdown("""
    <style>
    .custom-font {
        font-family: 'Courier New', Courier, monospace;
        font-size: 24px;
        color: #FF5733;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="custom-font">This is text with a custom font and size!</p>', unsafe_allow_html=True)

    # Add some styled text and lists
    st.markdown("### Styled List Example")
    st.markdown("""
    - <span style='color: red;'>Red Item</span>
    - <span style='color: orange;'>Orange Item</span>
    - <span style='color: yellow;'>Yellow Item</span>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="custom-font-2">Visually getting things on the page that look the way you desire is a good first step. You also need to make sure that you can insert things and link models or APIs. You can also make intuitive displays that reflect mental models. Everyone knows what a search bar does. Abstracting away some of the hard to grasp concepts give people an easy way to interact with your content and data.</p>', unsafe_allow_html=True)


    st.write("Enter a keyword and click the button to see an image based on your search term.")

    # User input for search term
    search_query = st.text_input("Enter a search term (e.g., 'nature', 'city', 'technology'): ")

    # Unsplash API URL and access key (replace 'YOUR_ACCESS_KEY' with your actual API key)
    unsplash_url = "https://api.unsplash.com/photos/random"
    access_key = ""

    # Button to trigger image generation
    if st.button("Generate Image"):
        # Make a request to the Unsplash API
        params = {
            "query": search_query,
            "client_id": access_key
        }
        response = requests.get(unsplash_url, params=params)
        
        if response.status_code == 200:
            # Extract the image URL
            data = response.json()
            image_url = data["urls"]["regular"]  # Gets the regular-sized image

            # Display the image
            st.image(image_url, caption=f"Image of {search_query}", use_container_width=True)
        else:
            st.write("An error occurred while fetching the image. Please try again.")

    st.divider()
    st.markdown('<p class="custom-font-2">Now this app has functionality to show progress but for your own applications you may want to add extra insights. This gives users peace of mind when they may be waiting for certain things to start.</p>', unsafe_allow_html=True)

    # Button to initiate the progress bar
    if st.button("Click Me!"):
        # Indicate that the button was clicked
        st.success("Button was clicked!")

        # Display a progress bar
        progress_bar = st.progress(0)

        # Simulate the progress
        for i in range(100):
            time.sleep(0.01)  # Sleep for a short duration to simulate loading time
            progress_bar.progress(i + 1)

        # Optionally show a message when done
        st.success("Progress Complete!")

        # After completion, reset the progress bar
        progress_bar.empty()  # This will clear the progress bar from the UI
    
    st.divider()
    st.markdown("[Visit the Streamlit documentation](https://docs.streamlit.io/) for more information on building apps.")

elif st.session_state.current_page == 'Data Viewer':
    
    st.markdown("""
    <style>
    .custom-font {
        font-family: 'Courier New', Courier, monospace;
        font-size: 24px;
        color: #FF3300;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color: red; font-size: 50px;'>Welcome to the Spotify Song Explorer Data View</h1>", unsafe_allow_html=True)

    st.markdown('<p class="custom-font">You can get a good grasp of the data on this page and view it quickly. Only the relevant columns are displayed but there are others.</p>', unsafe_allow_html=True)

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('merged.csv')

    data = load_data()
    filtered_data = data[["name","popularity","artists","release_date", "danceability","energy","loudness","tempo"]].rename(
        columns={"name":"Song","popularity":"Popularity","artists":"Artists","release_date":"Release Date","danceability":"Danceability","energy":"Energy",
                 "loudness":"Loudness","tempo":"Tempo"})
    st.dataframe(filtered_data, height=1000, width=1500)

elif st.session_state.current_page == 'Explore Songs':
    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('merged.csv')

    data = load_data()
    #Title and Description
    st.markdown("<h1 style='font-size: 65px; color: red;'>Spotify Song Explorer</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .custom-font {
        font-family: 'Courier New', Courier, monospace;
        font-size: 24px;
        color: #FF5733;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="custom-font">Welcome to the Spotify Song Explorer! This app is designed to showcase songs to users in a quick and intuitive way. Using data from 1921 to 2020 and over 600,000 tracks, we make it easy to explore, filter, and visualize song attributes such as popularity, tempo, genre, and mood. If you see anything you like you can download the .csv of the songs and export them for yourself. Let’s dive into the music!</p>', unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 45px; color: white;'>Popularity Explorer</h1>", unsafe_allow_html=True)
    st.write("Use the slider to adjust the popularity and you will see the name of the song and the popularity, release date, and tempo along with the artist/s who sang the song.")
    # Sidebar for simplicity and filtering options
    popularity = st.slider("Select Popularity", 0, 100, (50, 100))

    # Main filtered data display
    filtered_data = data[(data['popularity'] >= popularity[0]) & 
                        (data['popularity'] <= popularity[1])]
    st.write(f"Showing results for songs with popularity between {popularity[0]} and {popularity[1]}")
    st.dataframe(filtered_data[['name', 'artists', 'popularity', 'release_date', 'tempo']], width=1000)

    st.divider()
    st.markdown("<h1 style='font-size: 45px; color: white;'>Tempo Explorer</h1>", unsafe_allow_html=True)
    st.write("Use the slider to adjust the tempo and you will see the name of the song, the popularity, release date, and tempo along with the artist/s who sang the song.")
    # Sidebar for simplicity and filtering options
    # Efficiency: Additional filters
    tempo = st.slider("Filter by Tempo (BPM)", int(data['tempo'].min()), int(data['tempo'].max()))
    filtered_data = filtered_data[filtered_data['tempo'] >= tempo]
    st.write(f"Songs with tempo above {tempo} BPM")
    st.dataframe(filtered_data[['name', 'artists', 'popularity', 'tempo']], width=1000)

    # Intuitiveness: Download filtered data
    st.write("Download Filtered Data")
    st.download_button("Download CSV", filtered_data.to_csv(index=False), file_name="filtered_songs.csv")

elif st.session_state.current_page == 'Find a Song':
    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('merged.csv')

    data = load_data()
    #Title and Description
    st.markdown("<h1 style='font-size: 32px;'>Spotify Song Explorer</h1>", unsafe_allow_html=True)
    st.write("""
        Welcome to the Spotify Song Explorer! This app is designed to showcase songs to users in a quick and intuitive way.
        Using data from 1921 to 2020 and over 600,000 tracks, we make it easy to explore, filter, and visualize song attributes
        such as popularity, tempo, genre, and mood. Let’s dive into the music!
    """)

        # Supportive Design: Search box with helper text
    song_query = st.text_input("Search for a Song", help="Type a song name to quickly search through the dataset.")
    
    # Function to find the best fuzzy matches with progress tracking
    def find_best_matches_with_progress(query, df, limit=5):
        """
        Finds the best fuzzy matches in the 'name' column of the DataFrame based on the 'query' with a progress bar.
        Returns the top matches with scores.
        """
        stage_messages = ["Preparing search...", "Finding best matches...", "Finalizing results..."]
    
        # Display each stage message with a delay
        for stage in stage_messages:
            st.write(stage)
        
        # Extract names as a list for fuzzy matching
        names = data['name'].tolist()
        
        # Perform fuzzy matching to get the top matches
        best_matches = process.extract(query, names, limit=limit)
        
        
        # Filter DataFrame to only include rows that match the best results
        match_names = [match[0] for match in best_matches]
        filtered_df = data[data['name'].isin(match_names)]
        
        return filtered_df, best_matches

    if song_query:
        # Get the best matches in the "name" column with progress bar
        filtered_df, best_matches = find_best_matches_with_progress(song_query, data)

        # Display the filtered DataFrame and match scores
        if not filtered_df.empty:
            st.write("Top Matches:")
            for match_name, score in best_matches:
                st.write(f"{match_name} (Score: {score})")
            st.dataframe(filtered_df)
        else:
            st.write("No matches found.")

        
elif st.session_state.current_page == 'Popularity Model':
     # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('merged.csv')

    data = load_data()

    # Title and description
    st.markdown("<h1 style='font-size: 32px;'>Popularity Model</h1>", unsafe_allow_html=True)
    st.write("""
        The Spotify Song Explorer Popularity Model features an innovative machine learning model designed to predict the popularity of songs based on key audio characteristics such as danceability, 
             energy, and tempo. This model leverages a Random Forest Classifier trained on a comprehensive dataset containing over 600,000 tracks from 1921 to 2020. 
             By analyzing these features, the model classifies songs into discrete popularity levels, offering valuable insights into what makes a song resonate with listeners. 
             The dataset, enriched with attributes like artist, release date, and loudness, empowers users to explore and understand the factors influencing musical trends over the years. 
             This tool not only demonstrates the application of machine learning in music analytics but also serves as an educational example of data-driven decision-making.
    """)

    # Subset data (optional, 10%)
    st.markdown("#### Subsetting Data")
    subset_data = st.checkbox("Use a 10% subset of the data for faster performance")

    if subset_data:
        data = data.groupby('popularity', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

    # Features and target variable
    features = ["danceability", "energy", "tempo", "loudness", "duration_ms", "speechiness", "acousticness", "instrumentalness", "valence"]
    target = "popularity"

    # Handle missing values
    data = data.dropna(subset=features + [target])

    # Splitting the data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Train a machine learning model
    st.subheader("Train a Machine Learning Model")
    if st.button("Train Model"):
        model = RandomForestClassifier(n_jobs=-1, random_state=42)
        with st.spinner("Training the model... This might take a moment."):
            with tqdm(total=len(X_train)) as progress:
                model.fit(X_train, y_train)
                progress.update(len(X_train))

        # Save model to avoid retraining
        joblib.dump(model, 'random_forest_model.pkl')

        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)

        st.write(f"Accuracy: {accuracy:.2f}")

        # Convert the classification report into a pandas DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # Display the DataFrame as a styled table
        st.subheader("Classification Report")
        st.table(report_df)

        # Feature importance
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        for feature, importance in zip(features, importances):
            st.write(f"{feature}: {importance:.2f}")

elif st.session_state.current_page == "About Us":
    st.title("About the Spotify Song Explorer")

    st.write("""
        The Spotify Song Explorer is designed to offer users an interactive way to explore Spotify data.
        With access to song information from over 600,000 tracks dating from 1921 to 2020, this app allows
        users to filter, analyze, and visualize song characteristics like popularity, tempo, and mood.
    """)

    st.markdown("<h1 style='font-size: 32px;'>Built by Ben Johnson and Amir Saeed</h1>", unsafe_allow_html=True)