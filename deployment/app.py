# Import necessary libraries
import streamlit as st
import eda
import predict
import pandas as pd

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Navigation sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔍 Prediction"])

    if page == "🏠 Home":
        # Sidebar content for Home page
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 About the Models")
        
        # Sentiment Analysis Model
        st.sidebar.write("**Sentiment Analysis Model (BERT-based)**")
        sentiment_accuracy = 0.89
        st.sidebar.write("🎯 Model Accuracy:")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Accuracy", f"{sentiment_accuracy:.2%}")
        col2.metric("Error Rate", f"{1-sentiment_accuracy:.2%}")
        st.sidebar.write("Analyzes customer feedback to predict sentiment.")
        st.sidebar.write("**💡 What does this mean?**")
        st.sidebar.write("The model correctly classifies the sentiment of customer feedback 89% of the time. This high accuracy ensures that we can reliably interpret customer opinions and make informed decisions based on their feedback.")

        st.sidebar.markdown("---")

        # Churn Prediction Model
        st.sidebar.write("**Churn Prediction Model (SVC)**")
        churn_recall = 0.89
        st.sidebar.write("🎯 Model Recall:")
        st.sidebar.progress(churn_recall)
        st.sidebar.write(f"{churn_recall:.2%}")
        st.sidebar.write("**💡 What does this mean?**")
        st.sidebar.write("The model correctly identifies 89% of actual churning customers. This high recall minimizes false negatives, ensuring we catch most at-risk customers and can take proactive retention measures.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Fun Fact")
        st.sidebar.info("It costs 5-25 times more to acquire a new customer than it does to retain an existing one.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Tools Utilized")
        st.sidebar.write("""
        - `Streamlit` for web app development
        - `Pandas` for data manipulation
        - `Plotly Express` for interactive visualizations
        - `PyTorch` and `Transformers` for sentiment analysis (BERT)
        - `Scikit-learn` for machine learning models (SVC)
        - `Pickle` for model serialization
        """)

        # Main content for Home page
        st.title("🏃 Welcome to Customer Churn Prediction Tool")
        st.write("Empowering businesses with data-driven insights to retain customers and boost growth.")
        # Display image
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("predictix.jpg", caption="Predictix: Customer Churn Prediction", use_column_width=True)
        
        st.write("""
        `Predictix` is an **innovative app** designed to help businesses **understand and predict customer churn risk**. 
        Our application combines **powerful Exploratory Data Analysis (EDA)** with **advanced prediction capabilities**, 
        utilizing a **sophisticated two-step approach**. First, we employ a **BERT-based model** for **sentiment analysis**, 
        which accurately predicts sentiment from customer feedback. This sentiment data is then combined with other 
        customer information and fed into a **Support Vector Classifier (SVC)** to predict the **likelihood of churn**. 
        This **comprehensive approach** allows businesses to gain **deep insights** into customer behavior and take 
        **proactive measures** to improve retention. Whether you're looking to **explore your data** or **make predictions**, 
        Predictix has you covered. Simply use the **navigation pane** on the left to access the different modules and 
        start leveraging the power of **data-driven decision making** for your business.
        """)

        st.markdown("---")
        
        # Dataset information
        st.write("#### 📊 Dataset")
        st.info("""
        The dataset contains customer feedback used to predict sentiment, and then combines this sentiment analysis with customer information to predict customer churn.
        
        This two-step approach allows for a more nuanced understanding of customer behavior and improved churn prediction.
        
        Dataset source: [Florist Customer Churn](https://huggingface.co/datasets/iammkb2002/florist_customer_churn)
        """)

        # Checkbox to show/hide dataset column description
        if st.checkbox("Show dataset column description", value=True):
            st.table(pd.DataFrame({
                "Column Name": ["customer_id", "churn", "tenure", "monthly_charges", "total_charges", "contract", "payment_method", "feedback", "sentiment", "topic"],
                "Description": [
                    "Unique identifier for each customer",
                    "Indicates whether the customer has left (True/False)",
                    "Number of months the customer has been with the company",
                    "Amount charged to the customer monthly (in local currency)",
                    "Total amount charged to the customer over their tenure",
                    "Type of contract the customer has (e.g., one year, month-to-month, two year)",
                    "Payment method used by the customer (e.g., credit card, electronic check)",
                    "Customer feedback comments regarding the service or product",
                    "Sentiment of the feedback (positive/negative) - predicted by our BERT model",
                    "Topic category of the feedback (e.g., bouquet preferences, delivery issues, general feedback)"
                ]
            }))
        
    # Problem Statement
        st.write("#### ⚠️ Problem Statement")
        st.warning("""
        In today's competitive market, understanding customer sentiment and predicting churn are crucial for business success. 
        However, manually analyzing large volumes of customer feedback and identifying potential churners is time-consuming 
        and prone to human error. Predictix addresses these challenges by automating both the sentiment analysis process 
        and churn prediction, allowing businesses to respond promptly to customer needs and preferences.
        
        Customer churn is a significant challenge for businesses, leading to revenue loss and increased 
        acquisition costs. Early identification of customers likely to churn is crucial for implementing 
        effective retention strategies. As a data scientist, your task is to develop a machine learning 
        model that can predict customer churn based on historical data, customer behavior patterns, and sentiment analysis.
        
        The goal is to develop a two-step model approach with high accuracy and recall to identify potential churners, 
        allowing the business to take proactive measures to retain these customers.
    """)
        
        # Project Objective
        st.write("#### 🎯 Objective")
        st.success("""
        This project aims to create a two-step classification model to predict customer churn:
        1. Use a BERT-based model to analyze customer feedback and predict sentiment.
        2. Use an SVC model to predict customer churn based on the predicted sentiment and other customer information.
        
        Model performance will be primarily assessed using Accuracy for the sentiment analysis model and 
        Recall for the churn prediction model to measure effectiveness in identifying potential churners, 
        minimizing the risk of missing customers who are likely to leave.
        """)

    elif page == "📊 EDA":
        # Run the EDA module
        eda.run()
    
    elif page == "🔍 Prediction":
        # Run the Prediction module
        predict.run()

if __name__ == "__main__":
    main()