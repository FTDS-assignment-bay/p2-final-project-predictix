# Import necessary libraries
import streamlit as st  
import pandas as pd  
import plotly.express as px  
import numpy as np 

def run():
    # Set the title of the Streamlit app
    st.title('📊 Exploratory Data Analysis')
    st.write('---')

    # Load the dataset from a CSV file
    df = pd.read_csv('florist_customer_churn_raw_fix_cleaned.csv')

    # Display the first few rows of the dataset
    st.subheader('📂 Dataset Overview: ')
    st.dataframe(df.head(10))

    # -- CONTAINER --
    # Creating container for home-page description
    ins_total_churn = st.container()
    ins_total_churn.markdown('<h1 style="font-size: 30px;">🧠 Quick to Know about Dataset: </h1>', unsafe_allow_html=True)
    ins_total_churn.write("The dataset contains various customer behavior indicators that may be associated with **customer churn**. From this data, our team will provide a classification based on the sentiment from the `feedback` to predict whether a customer will churn or not.")
    st.write('---')

    # -- DATA EXPLORATION --
    st.subheader('🗺️ Data Exploration')

    # --1. CHURN PIE CHART VIZ ---
    # Count the number of True/False in 'churn' column
    churn_count = df['churn'].value_counts()

    # Create a pie chart using Plotly with a purple color palette
    fig = px.pie(values=churn_count.values, 
                 names=churn_count.index, 
                 title="Total Churn Pie Chart Distribution",
                 color_discrete_sequence=px.colors.sequential.Purples_r) 

    # Show the chart in Streamlit
    st.subheader("🏃 Total Churn")
    st.plotly_chart(fig)

    # -- INSIGHT TOTAL CHURN --
    ins_total_churn = st.container()
    ins_total_churn.markdown('<h1 style="font-size: 30px;">💭 Insight: </h1>', unsafe_allow_html=True)
    ins_total_churn.write("""
                        - A nearly equal split of `true` and `false` churn indicates that about half of customers remain `loyal` and `the other half churn`.
                        - Churn ratio approaching `50-50` indicates that there is a significant risk of losing customers.
                        - This indicates that we should focus on customer retention strategies and service improvements to `reduce true churn and maintain customer loyalty`."""
                          )

    # -- 2. POSITIF FEEDBACK --
    # Membuat chart menggunakan Plotly Express
    # Filter for rows where sentiment is 'positive'
    positive_df = df[df['sentiment'] == 'positive']

    # Group the data by 'topic' and count the occurrences
    positive_topic_counts = positive_df['topic'].value_counts().reset_index()
    positive_topic_counts.columns = ['Topic', 'Count of Sentiment']

    # Create the bar chart using Plotly
    fig = px.bar(positive_topic_counts, 
                 x='Count of Sentiment', 
                 y='Topic', 
                 orientation='h', 
                 color_discrete_sequence=['#8a2be2'],  # Purple color
                 title="Positive Sentiment by Topic")

    # Display the plot in Streamlit
    st.subheader("🗣️💬 Positive Sentiment by Topic")
    st.plotly_chart(fig)

    # -- INSIGHT POSITIF FEEDBACK BY TOPIC --
    ins_positive_feedback = st.container()
    ins_positive_feedback.markdown('<h1 style="font-size: 30px;">💭 Insight: </h1>', unsafe_allow_html=True)
    ins_positive_feedback.write("""
                                - `Product Quality` (Kualitas Produk) receives the most attention in positive sentiment, with more than 100 people expressing satisfaction with the product.
                                - `General Feedback` is also quite high, indicating that many customers provide good general feedback regarding the service or product.
                                - `Bouquet Preferences` also has a substantial amount of positive sentiment, indicating that customers are quite satisfied with the available flower arrangement options.
                                - `Customer Service` receives positive sentiment, although not as high as some other topics, but it shows that customer service is still fairly appreciated.
                                - `Price Appreciation` (Apresiasi Harga) shows that some customers feel the offered prices are quite reasonable.
                                - `Delivery Quality` (Kualitas Pengiriman) and `Delivery Issues` (Masalah Pengiriman) are relatively low in positive sentiment, meaning the delivery aspect is not a major strength.
                                """)
    st.write('---')


    # -- 3. NEGATIF FEEDBACK --
    # Filter for rows where sentiment is 'negative'
    negative_df = df[df['sentiment'] == 'negative']

    # Group the data by 'topic' and count the occurrences
    negative_topic_counts = negative_df['topic'].value_counts().reset_index()
    negative_topic_counts.columns = ['Topic', 'Count of Sentiment']

    # Create the bar chart using Plotly
    fig = px.bar(negative_topic_counts, 
                 x='Count of Sentiment', 
                 y='Topic', 
                 orientation='h', 
                 color_discrete_sequence=['#8a2be2'],  # Purple color
                 title="Negative Sentiment by Topic")

    # Display the plot in Streamlit
    st.subheader("🗣️💬 Negative Sentiment by Topic")
    st.plotly_chart(fig)

    # -- INSIGHT NEGATIF FEEDBACK BY TOPIC --
    ins_negative_feedback = st.container()
    ins_negative_feedback.markdown('<h1 style="font-size: 30px;">💭 Insight: </h1>', unsafe_allow_html=True)
    ins_negative_feedback.write("""
                                - `Product Quality` is also a major topic in negative sentiment, with more than 140 negative comments. This indicates that while there is a lot of praise, there are also significant complaints about product quality.
                                - `Price Complaints` is a major negative topic, meaning many customers feel that the prices offered are too high or not meeting their expectations.
                                - `Delivery Issues` is also a major problem in negative sentiment, showing that delivery is a primary source of complaints.
                                - `Bouquet Preferences` also has a fair amount of negative sentiment, indicating that while many are satisfied, there are also those who feel the flower arrangements do not meet their expectations.
                                - `Customer Service` has received some negative sentiment, though it is not as prominent as other topics.
                                - `Delivery Quality` has very minimal negative sentiment, indicating that the quality of delivery is less frequently complained about compared to delivery issues overall.
                                """)
    st.write('---')

    # -- 4. CHURN RATE --
    # Group the data by 'churn' and 'contract' and count the occurrences
    # Map the churn column to categorical values: False -> 'Not Churned', True -> 'Churned'
    df['churn_category'] = df['churn'].map({False: 'Not Churned', True: 'Churned'})

    # Group the data by 'churn_category' and 'contract' and count the occurrences
    churn_contract_counts = df.groupby(['churn_category', 'contract']).size().reset_index(name='Count of Churn')

    # Create the bar chart using Plotly
    fig = px.bar(churn_contract_counts, 
                 x='Count of Churn', 
                 y='contract', 
                 color='churn_category',  # Use the new categorical churn column
                 barmode='group', 
                 orientation='h',
                 color_discrete_sequence=['#8a2be2', '#c8a2c8'],  # Purple color shades
                 title="Churn Rate by Contract Type")

    # Display the plot in Streamlit
    st.subheader("🏃  or  🙆  by Contract Type")
    st.plotly_chart(fig)

    # -- INSIGHT CHURN RATE BY CONTRACT TYPE --
    ins_churn_rate = st.container()
    ins_churn_rate.markdown('<h1 style="font-size: 30px;">💭 Insight: </h1>', unsafe_allow_html=True)
    ins_churn_rate.write("""
                        - `Short-term (monthly)` contracts have a very high churn rate, indicating that customers tend to leave the service more easily if they are not tied to a long-term contract
                        - `Long-term contracts` (one and two years) are more effective in retaining customers than short-term contracts.
                         """)
    st.write('---')

    st.write(
        '<p style="font-size: 15px; text-align: center;">All Rights Reserved | Made by ❤️</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run()