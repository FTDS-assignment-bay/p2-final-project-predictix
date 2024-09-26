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

    st.write("## ✍🏻📑 Input Data: ")
    st.write("")  # Add a blank line for spacing
    with st.form(key="data"):
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID")
            tenure = st.number_input("Tenure")
        with col2:
            contract = st.selectbox(
                "contract", ['one year', 'month-to-month', 'two year']
            )
            payment_method = st.selectbox(
                "payment_method", ['credit card', 'electronic check', 'bank transfer', 'mailed check']
            )
        monthly_charges = st.number_input("Monthly charges")
        total_charges = st.number_input("Total charges")
        feedback = st.text_area("Feedback")
        topic = st.selectbox(
            "topic", ['bouquet preferences', 'delivery issues', 'general feedback', 'price complaints', 'delivery quality', 'product quality', 'customer service', 'price appreciation']
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
