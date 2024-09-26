# Import Libraries
import streamlit as st
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

# Load the tokenizer and model for sentiment analysis
model_dir = './saved_model/'  # Update this path if your model is saved elsewhere

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Function to perform sentiment analysis
def predict_sentiment(texts):
    # Tokenize and encode the texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

    # Map predictions to labels
    label_map = {0: 'Negative', 1: 'Positive'}
    predicted_labels = [label_map[pred.item()] for pred in predicted_classes]
    confidences = confidences.cpu().numpy()

    return predicted_labels, confidences

st.write('')

def run():
    st.title("🏃 Customer Churn Prediction")
    st.markdown('---')

    st.sidebar.markdown("---")
    
    st.sidebar.write("### BERT Model")
    st.sidebar.write("This application uses a BERT model to analyze the sentiment of customer feedback. BERT is highly effective at understanding context and nuances in text, even for long feedback, despite being trained on shorter sentences in our dataset.")
    st.sidebar.image("bert.jpg", width=200)
    st.sidebar.write("BERT (Bidirectional Encoder Representations from Transformers) works by understanding the context of words in a sentence by looking at the words that come before and after them. This bidirectional approach allows BERT to capture the meaning of words more accurately.")
    
    st.sidebar.markdown('---')  # Line separating between models
    
    st.sidebar.write("### SVM Classifier")
    st.sidebar.write("After sentiment analysis, an SVM classifier is used to predict whether a customer is likely to churn based on the sentiment and other features provided.")
    st.sidebar.image("svm.png", width=200)
    st.sidebar.write("SVM (Support Vector Machine) works by finding the hyperplane that best separates the data into different classes. In this case, it uses the sentiment and other features to predict customer churn.")

    st.write("## ✍🏻📑 Input Data: ")
    st.write("")  # Add a blank line for spacing
    st.write("This prediction will be done by first analyzing the sentiment from the feedback provided using a BERT model. The sentiment, along with other features, will then be used to classify whether the customer is likely to churn or not churn using an SVM classifier.")
    with st.form(key="data"):
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID", value="CUST001")
            tenure = st.number_input("Tenure (in months)", value=12)
        with col2:
            contract = st.selectbox(
                "contract", ['one year', 'month-to-month', 'two year'], index=1
            )
            payment_method = st.selectbox(
                "payment_method", ['credit card', 'electronic check', 'bank transfer', 'mailed check'],index=0
            )
        monthly_charges = st.number_input("Monthly charges (in $)",step=1.00)
        total_charges = st.number_input("Total charges (in $)",step=1.00)
        feedback = st.text_area(
            "Feedback", 
            value="As an event planner, I've worked with many florists, but this one stands out from the crowd. For a recent high-profile corporate gala, they provided centerpieces that were nothing short of spectacular. The creativity and artistry in their designs elevated the entire event. They listened carefully to our theme and color scheme, then created arrangements that perfectly captured the essence of the evening. The flowers were fresh, vibrant, and lasted throughout the night, even under bright lights and air conditioning. What impressed me most was their flexibility - when we needed last-minute changes due to a shift in table arrangements, they accommodated without hesitation. Their team was professional, punctual, and a joy to work with. They arrived well before the event to set up and stayed until everything was perfect. The value for money was excellent, considering the high-quality results and the level of service provided. I've already booked them for several upcoming events, including a charity fundraiser and a product launch. It's rare to find a vendor that consistently exceeds expectations, but this florist does just that.",
            height=300
        )
        topic = st.selectbox(
            "topic", ['bouquet preferences', 'delivery issues', 'general feedback', 'price complaints', 'delivery quality', 'product quality', 'customer service', 'price appreciation'], index=2
        )
        # Submit button
        submit = st.form_submit_button("🔘 Predict")

    if submit:
        data = {
            "customer_id": customer_id,
            "tenure": tenure,
            "contract": contract,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "feedback": feedback,
            "topic": topic
        }

        # Sentiment Analysis
        if feedback.strip() != "":
            st.write("### 📊 Sentiment Analysis")
            st.markdown('---')
            labels, confidences = predict_sentiment([feedback])
            label = labels[0]
            confidence = confidences[0]
            st.success(f"Predicted Sentiment: **{label}**")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            data['sentiment'] = label
        else:
            st.warning("Feedback is empty, skipping sentiment analysis.")
            data['sentiment'] = None

        # Convert data to DataFrame for churn prediction
        data = pd.DataFrame([data])

        # Load customer churn model
        with open('model.pkl', 'rb') as file_1:
            classification = pickle.load(file_1)

        # Predict customer churn using the loaded model
        churn = classification.predict(data)

        # Display churn prediction result
        st.write("### 🕵️‍♂️ Prediction Results: ")
        st.markdown('---')
        st.write("")  # Add a blank line for spacing
        
        if churn == False:
            st.error("🏃 **Customer is Gonna Churn!!**")
            st.image('8908ec58-057d-4bfe-9ce5-74322486859a.png')
            st.write('') #space
            st.write('### 💭 Feedback To Marketing Team: ')
            st.error("""
                    - **The customer** is likely to churn if we don't take immediate steps to improve our **service quality**. Consistently providing subpar service will push them to seek out competitors who can meet their expectations more reliably, resulting in a loss of **long-term loyalty and revenue**.

                    - Without implementing a more effective **retention strategy**, our customers are at high risk of churning. It’s crucial that we personalize our **offerings, provide timely incentives, and enhance customer communication** to keep them **engaged and loyal** to our brand.

                    - The recent product changes have led to growing dissatisfaction among customers, increasing the likelihood of churn. To prevent this, we must swiftly **address their concerns, re-evaluate the changes, and ensure that future updates align with customer expectations** to regain their trust.

                    - If we fail to address customer concerns promptly, we risk driving them away. Providing timely and empathetic responses is essential to **resolving issues, maintaining trust, and ensuring that customers feel valued**, which is critical to reducing churn.

                    - Without a personalized engagement approach, customers are more likely to churn, as they will **feel disconnected** from our brand. Tailoring our **communication and offers** based on **individual customer preference**s can significantly improve **satisfaction and foster long-term loyalty**.
                     """)
        else:
            st.success("🙆 Customer is Not Gonna Churn")
            st.image('sss.png')
            st.balloons()

if __name__ == "__main__":
    run()