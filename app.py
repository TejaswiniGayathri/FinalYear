import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import base64


# Define a dictionary for user authentication (replace with a secure method in production)
user_credentials = {
    'username': 'teju',
    'password': 'teju'
}

# Function to check if the entered credentials are valid
def authenticate(username, password):
    return username == user_credentials['username'] and password == user_credentials['password']

# Function to generate user analytics graph
def user_analytics(selected_user, df):
    # Add your user analytics code here
    # For example, let's create a bar chart of message count per user
    user_message_counts = df.groupby('user').count()['message']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(user_message_counts.index, user_message_counts.values, color='green')
    plt.xlabel('User')
    plt.ylabel('Message Count')
    plt.title(f'Message Count per User ({selected_user})')
    return fig

def perform_topic_modeling(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Preprocess the text data (you may need to adjust this based on your preprocessing)
    text_data = df['message']

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_data = tfidf_vectorizer.fit_transform(text_data)

    # Perform Latent Dirichlet Allocation (LDA)
    num_topics = 5  # You can adjust the number of topics as needed
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(tfidf_data)

    # Extract and display the top words for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]

    return topics

# Define function for message clustering
def perform_message_clustering(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Preprocess the text data (you may need to adjust this based on your preprocessing)
    text_data = df['message']

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_data = tfidf_vectorizer.fit_transform(text_data)

    # Perform clustering using K-means
    num_clusters = 3  # You can adjust the number of clusters as needed
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_model.fit(tfidf_data)

    # Evaluate clustering quality
    silhouette_avg = silhouette_score(tfidf_data, kmeans_model.labels_)
    cluster_centers = kmeans_model.cluster_centers_

    # Assign cluster labels to messages
    df['cluster_label'] = kmeans_model.labels_

    return df, silhouette_avg, cluster_centers

# Add title and image to the page
st.title("WhatsApp Chat Analyzer")
image = Image.open("image.png")  # Replace with the path to your image
st.image(image, caption="Your Image Caption", use_column_width=True)

# Add a separator for better organization
st.markdown("---")

# User Authentication Section
login_status = False

# Check if the user is authenticated
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# If the user is not authenticated, show login form
if not st.session_state.authenticated:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
        else:
            st.sidebar.error("Invalid username or password")

# Main application content
if st.session_state.authenticated:
    st.sidebar.title("Whatsapp Chat Analyzer")

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # fetch unique users
        user_list = df['user'].unique().tolist()

        # Check if 'group_notification' is in the list before removing it
        if 'group_notification' in user_list:
            user_list.remove('group_notification')

        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

        if st.sidebar.button("Show Analysis"):
            # Stats Area
            st.markdown("# Extracted DataFrame from the GroupChat")
            st.write(df.head())

            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

            st.markdown("---")
            st.title("Top Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)

            with col2:
                st.header("Total Words")
                st.title(words)

            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)

            with col4:
                st.header("Links Shared")
                st.title(num_links)

            with col5:
                st.header("Number of Users")
                st.title(len(user_list) - 1)

            # Monthly timeline
            st.markdown("---")
            st.title("Monthly Timeline Analysis")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily timeline
            st.markdown("---")
            st.title("Daily Timeline Analysis")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Activity map
            st.markdown("---")
            st.title('Activity Map Analysis')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most Busy Day Analysis")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most Busy Month Analysis")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Weekly Activity Map
            st.markdown("---")
            st.title("Weekly Activity Map Analysis")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # Finding the busiest users in the group (Group level)
            if selected_user == 'Overall':
                st.markdown("---")
                st.title('Most Busy Users Analysis')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots(figsize=(15, 7))

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

            # WordCloud
            st.markdown("---")
            st.title("Wordcloud Analysis")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(df_wc)
            st.pyplot(fig)

            # Most common words
            st.markdown("---")
            st.title('Most Common Words Analysis')
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots(figsize=(15, 7))
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.title('Most common words')
            st.pyplot(fig)

            # Emoji analysis
            st.markdown("---")
            st.title("Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)

            # Sentiment Analysis
            st.markdown("---")
            st.title("Sentiment Analysis")
            sentiment_df = helper.sentiment_analysis(selected_user, df)
            st.dataframe(sentiment_df)

            # Conversation Flow Analysis
            st.markdown("---")
            st.title("Conversation Flow Analysis")
            conversation_flow_df = helper.conversation_flow_analysis(selected_user, df)
            st.dataframe(conversation_flow_df)

            # User Analytics
            st.markdown("---")
            st.title("User Analytics")
            user_analytics_fig = user_analytics(selected_user, df)
            st.pyplot(user_analytics_fig)


# Logout button (displayed only when authenticated)
if st.session_state.authenticated and st.sidebar.button("Logout"):
    st.session_state.authenticated = False
