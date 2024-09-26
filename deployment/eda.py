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

    # Sidebar content
    st.sidebar.title("EDA Options")
    
    # Add a selectbox for choosing analysis type
    analysis_option = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Dataset Overview", "Churn Distribution", "Sentiment Analysis", "Contract Analysis", "Word Cloud"]
    )
    
    # Add a slider for sample size in Dataset Overview
    if analysis_option == "Dataset Overview":
        sample_size = st.sidebar.slider("Sample size", min_value=5, max_value=50, value=10, step=5)
    
    # Add radio buttons for sentiment type
    if analysis_option == "Sentiment Analysis":
        sentiment_option = st.sidebar.radio("Choose sentiment to display:", ("Positive", "Negative"))
    
    # Add radio buttons for word cloud type
    if analysis_option == "Word Cloud":
        wordcloud_option = st.sidebar.radio("Choose word cloud to display:", ("Positive Sentiment", "Negative Sentiment"))
    
    # Add checkbox for showing statistics in Feature Explorer
    show_stats = st.sidebar.checkbox("Show feature statistics", value=True)

    # Add more content to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Key Features")
    st.sidebar.write("""
    - Interactive visualizations
    - Sentiment analysis insights
    - Churn distribution analysis
    - Contract type impact on churn
    - Word cloud for sentiment analysis
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Tools Utilized")
    st.sidebar.write("""
    - `Streamlit` for web app development
    - `Pandas` for data manipulation
    - `Plotly Express` for interactive visualizations
    - `NumPy` for numerical operations
    """)

    st.sidebar.markdown("---")
    st.sidebar.info("Explore different aspects of the customer churn data using the options above.")

    # Main page content
    st.write("Welcome to the EDA page. Choose an analysis to explore:")

    if analysis_option == "Dataset Overview":
        st.subheader('📂 Dataset Overview: ')
        
        # Move multi-select for choosing columns to display to main page
        columns_to_display = st.multiselect(
            "Select columns to display",
            options=list(df.columns),
            default=list(df.columns)
        )
        
        st.dataframe(df[columns_to_display].head(sample_size))

        st.markdown('<h3 style="font-size: 24px;">🧠 Quick to Know about Dataset: </h3>', unsafe_allow_html=True)
        st.write("The dataset contains various customer behavior indicators that may be associated with **customer churn**. From this data, our team will provide a classification based on the sentiment from the `feedback` to predict whether a customer will churn or not.")

    elif analysis_option == "Churn Distribution":
        st.subheader("🏃 Total Churn")
        churn_count = df['churn'].value_counts()
        fig = px.pie(values=churn_count.values, 
                     names=churn_count.index, 
                     title="Total Churn Pie Chart Distribution",
                     color_discrete_sequence=px.colors.sequential.Purples_r) 
        st.plotly_chart(fig)

        st.markdown('<h3 style="font-size: 24px;">💭 Insight: </h3>', unsafe_allow_html=True)
        st.write("""
                - A nearly equal split of `true` and `false` churn indicates that about half of customers remain `loyal` and `the other half churn`.
                - Churn ratio approaching `50-50` indicates that there is a significant risk of losing customers.
                - This indicates that we should focus on customer retention strategies and service improvements to `reduce true churn and maintain customer loyalty`."""
                  )

    elif analysis_option == "Sentiment Analysis":
        st.subheader("🗣️💬 Sentiment Analysis")

        if sentiment_option == "Positive":
            positive_df = df[df['sentiment'] == 'positive']
            positive_topic_counts = positive_df['topic'].value_counts().reset_index()
            positive_topic_counts.columns = ['Topic', 'Count of Sentiment']
            fig = px.bar(positive_topic_counts, 
                         x='Count of Sentiment', 
                         y='Topic', 
                         orientation='h', 
                         color_discrete_sequence=['#8a2be2'],
                         title="Positive Sentiment by Topic")
            st.plotly_chart(fig)
        else:
            negative_df = df[df['sentiment'] == 'negative']
            negative_topic_counts = negative_df['topic'].value_counts().reset_index()
            negative_topic_counts.columns = ['Topic', 'Count of Sentiment']
            fig = px.bar(negative_topic_counts, 
                         x='Count of Sentiment', 
                         y='Topic', 
                         orientation='h', 
                         color_discrete_sequence=['#8a2be2'],
                         title="Negative Sentiment by Topic")
            st.plotly_chart(fig)

        st.markdown('<h3 style="font-size: 24px;">💭 Insight: </h3>', unsafe_allow_html=True)
        if sentiment_option == "Positive":
            st.write("""
                    - `Product Quality` receives the most attention in positive sentiment, with more than 100 people expressing satisfaction with the product.
                    - `General Feedback` is also quite high, indicating that many customers provide good general feedback regarding the service or product.
                    - `Bouquet Preferences` also has a substantial amount of positive sentiment, indicating that customers are quite satisfied with the available flower arrangement options.
                    - `Customer Service` receives positive sentiment, although not as high as some other topics, but it shows that customer service is still fairly appreciated.
                    - `Price Appreciation` shows that some customers feel the offered prices are quite reasonable.
                    - `Delivery Quality` and `Delivery Issues` are relatively low in positive sentiment, meaning the delivery aspect is not a major strength.
                    """)
        else:
            st.write("""
                    - `Product Quality` is also a major topic in negative sentiment, with more than 140 negative comments. This indicates that while there is a lot of praise, there are also significant complaints about product quality.
                    - `Price Complaints` is a major negative topic, meaning many customers feel that the prices offered are too high or not meeting their expectations.
                    - `Delivery Issues` is also a major problem in negative sentiment, showing that delivery is a primary source of complaints.
                    - `Bouquet Preferences` also has a fair amount of negative sentiment, indicating that while many are satisfied, there are also those who feel the flower arrangements do not meet their expectations.
                    - `Customer Service` has received some negative sentiment, though it is not as prominent as other topics.
                    - `Delivery Quality` has very minimal negative sentiment, indicating that the quality of delivery is less frequently complained about compared to delivery issues overall.
                    """)

    elif analysis_option == "Contract Analysis":
        st.subheader("🏃  or  🙆  by Contract Type")
        df['churn_category'] = df['churn'].map({False: 'Not Churned', True: 'Churned'})
        churn_contract_counts = df.groupby(['churn_category', 'contract']).size().reset_index(name='Count of Churn')
        fig = px.bar(churn_contract_counts, 
                     x='Count of Churn', 
                     y='contract', 
                     color='churn_category',
                     barmode='group', 
                     orientation='h',
                     color_discrete_sequence=['#8a2be2', '#c8a2c8'],
                     title="Churn Rate by Contract Type")
        st.plotly_chart(fig)

        st.markdown('<h3 style="font-size: 24px;">💭 Insight: </h3>', unsafe_allow_html=True)
        st.write("""
                - `Short-term (monthly)` contracts have a very high churn rate, indicating that customers tend to leave the service more easily if they are not tied to a long-term contract
                - `Long-term contracts` (one and two years) are more effective in retaining customers than short-term contracts.
                 """)

    elif analysis_option == "Word Cloud":
        st.subheader("☁️ Word Cloud")
        if wordcloud_option == "Positive Sentiment":
            st.image("wordcloud_positive.png", caption="Word Cloud for Positive Sentiment", use_column_width=True, width=150)
            st.markdown('<h3 style="font-size: 24px;">💭 Insight: </h3>', unsafe_allow_html=True)
            st.write("""
            1. **Frequent Mention of `Bouquet` and `Flowers`**: The words `bouquet` and `flowers` are prominently featured, indicating that customers often appreciate the quality and variety of the floral arrangements provided.
            2. **Emphasis on `Service`**: The word `service` appears frequently, suggesting that customers are generally satisfied with the level of service they receive.
            3. **Positive Adjectives**: Words like `always,` `happy,` `satisfied,` `quality,` and `great` are commonly used, reflecting a high level of customer satisfaction and positive experiences.
            4. **Subscription Model**: The word `subscription` is also notable, indicating that customers value the subscription service offered, which likely contributes to their positive feedback.
            5. **Consistency and Reliability**: Terms such as `always,` `every month,` and `arrive` suggest that customers appreciate the consistency and reliability of the service.

            Overall, the word cloud highlights the aspects of the service that customers find most appealing, such as the quality of the bouquets, the reliability of the service, and the positive experiences associated with the subscription model. These insights can help the company understand what they are doing well and continue to focus on these strengths.
            """)
        else:
            st.image("wordcloud_negative.png", caption="Word Cloud for Negative Sentiment", use_column_width=True, width=150)
            st.markdown('<h3 style="font-size: 24px;">💭 Insight: </h3>', unsafe_allow_html=True)
            st.write("""
            1. **Concerns About `Quality` and `Flowers`**: The words `quality` and `flowers` are prominently featured, indicating that many customers have concerns about the quality of the flowers they receive.
            2. **Issues with `Delivery`**: The word `delivery` appears frequently, suggesting that delivery-related issues are a common source of dissatisfaction among customers.
            3. **`Expensive` and `Costly`**: Terms like `expensive` and `costly` are notable, indicating that some customers feel the service or products are overpriced.
            4. **`Bouquet Size` and `Variety`**: Words such as `bouquet size` and `variety` suggest that customers are not satisfied with the size of the bouquets or the variety of flowers offered.
            5. **`Disappointed` and `Expected Better`**: The presence of words like `disappointed` and `expected better` reflects unmet expectations and general dissatisfaction with the service or product.
            6. **Mixed Sentiment on `Satisfied`**: Interestingly, the word `satisfied` appears in the negative feedback, possibly indicating that some customers are expressing conditional satisfaction or comparing their current experience to previous, more positive experiences.

            Overall, the word cloud highlights several areas for improvement, including flower quality, delivery service, pricing, bouquet size, and variety. Addressing these issues can help the company enhance customer satisfaction and reduce negative feedback.
            """)

    # Feature Explorer
    if show_stats:
        st.subheader("🔍 Feature Explorer")
        selected_feature = st.selectbox("Select a feature to explore", df.columns)
        if selected_feature:
            st.write(f"Statistics for {selected_feature}:")
            st.write(df[selected_feature].describe())

    st.write('---')
    st.write(
        '<p style="font-size: 15px; text-align: center;">All Rights Reserved | Made by ❤️</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run()