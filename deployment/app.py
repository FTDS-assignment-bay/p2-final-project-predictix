# Import necessary libraries
import streamlit as st
import eda
import predict

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
        st.sidebar.subheader("📊 About the Model")
        recall = 0.89  # You may want to update this value based on your model's performance
        st.sidebar.write("🎯 Model Recall:")
        st.sidebar.progress(recall)
        st.sidebar.write(f"{recall:.2%}")
        st.sidebar.write("**🤔 What is Recall?**")
        st.sidebar.write("Recall measures the model's ability to correctly identify positive cases (churning customers) out of all actual positive cases.")
        st.sidebar.write("**💡 What does this mean?**")
        st.sidebar.write("Out of all the customers who actually churn, our model correctly identifies 89% of them.")
        st.sidebar.write("This helps us minimize false negatives, *reducing the risk of overlooking customers who are likely to churn but weren't identified*")

        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Fun Fact")
        st.sidebar.info("It costs 5-25 times more to acquire a new customer than it does to retain an existing one.")

        # Main content for Home page
        st.title("🏃 Welcome to Customer Churn Prediction Tool")
        st.write("""
        This application provides functionalities for Exploratory Data Analysis and 
        Prediction regarding customer churn risk. Use the navigation pane on the left to 
        select the module you wish to utilize.
        """)
        
        # Display image
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("predictix.jpg", caption="Customer Churn Prediction", use_column_width=True)
        
        st.markdown("---")
        
        # Dataset information
        st.write("#### 📊 Dataset")
        st.info("""
        The dataset contains customer information including tenure, contract type, payment method,
        monthly charges, total charges, and feedback. It's used to predict customer churn.
        """)
        
        # Problem Statement
        st.write("#### ⚠️ Problem Statement")
        st.warning("""
        Customer churn is a significant challenge for businesses, leading to revenue loss and increased 
        acquisition costs. Early identification of customers likely to churn is crucial for implementing 
        effective retention strategies. As a data scientist, your task is to develop a machine learning 
        model that can predict customer churn based on historical data and customer behavior patterns.
        
        The goal is to develop a model with high recall to identify potential churners, allowing the 
        business to take proactive measures to retain these customers.
        """)
        
        # Project Objective
        st.write("#### 🎯 Objective")
        st.success("""
        This project aims to create a classification model to predict customer churn by evaluating 
        various algorithms. Model performance will be primarily assessed using Recall to measure 
        effectiveness in identifying potential churners, minimizing the risk of missing customers 
        who are likely to leave.
        """)

    elif page == "📊 EDA":
        # Run the EDA module
        eda.run()
    
    elif page == "🔍 Prediction":
        # Run the Prediction module
        predict.run()

if __name__ == "__main__":
    main()