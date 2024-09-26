# Predictix: Predicting Customer Churn using NLP and Machine Learning

<img src="./predictix.jpg" alt="Predictix Logo">

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;">Dataset</th>
        <th style="padding: 10px;">Hugging Face</th>
    </tr>
    <tr>
        <td style="padding: 10px;">
            <a href="https://huggingface.co/datasets/iammkb2002/florist_customer_churn">
                <img src="https://img.shields.io/badge/Dataset-Florist%20Customer%20Churn-orange" alt="Dataset">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://huggingface.co/spaces/amariayudha/Predictix">
                <img src="https://img.shields.io/badge/Demo-Hugging%20Face-blue" alt="Hugging Face Demo">
            </a>
        </td>
    </tr>
</table>

[Installation](#installation) | [Getting Started](#getting-started) | [Model Architecture](#model-architecture) | [Performance](#performance) | [Contributing](#contributing)

## Introduction

Predictix is an app designed to predict the likelihood of customer churn, with the ultimate goal of reducing churn rates and increasing customer satisfaction. By employing Natural Language Processing (NLP), this app will analyze customer feedback and perform classification to predict whether customers will switch to other services or remain loyal customers.

### Problem Statement

In today's competitive market, understanding customer sentiment is crucial for business success. However, manually analyzing large volumes of customer feedback is time-consuming and prone to human error. Predictix addresses this challenge by automating the sentiment analysis process, allowing businesses to respond promptly to customer needs and preferences. Then, predictix will predict whether the customer will churn or not using machine learning algorithm based on the sentiment, and other features.

### Objectives

1. Understand customer sentiment through their feedback
2. Build a model to classify sentiment in customer reviews
3. Use sentiment analysis to predict customer churn

### Proposed Solution

Predictix utilizes advanced NLP algorithms, including BERT, to preprocess and analyze customer feedback data. The system employs techniques such as tokenization, stopword removal, and sentiment classification to provide accurate and timely insights into customer sentiments.

## Installation

### Dependencies

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.10+
- Pandas
- NumPy
- Scikit-learn

To install Predictix:
```bash
pip install predictix
```

## Getting Started

```python
from predictix import SentimentAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment
feedback = "The bouquet was absolutely stunning! Fresh flowers and beautiful arrangement."
sentiment = analyzer.predict(feedback)

print(f"Sentiment: {sentiment}")
```

## Model Architecture

Predictix uses a fine-tuned BERT model for sentiment analysis. The architecture includes:

- BERT base model
- Custom classification head
- Fine-tuning on florist-specific dataset

## Performance

Our BERT-based model achieves:

- Train Accuracy: 92%
- Validation Accuracy: 94%
- Test Accuracy: 89%

| Model          | Train Accuracy | Validation Accuracy | Test Accuracy |
|----------------|----------------|---------------------|---------------|
| Random Forest  | 92%            | 91%                 | 87%           |
| LSTM           | 92%            | 92%                 | 89%           |
| BERT           | 92%            | 94%                 | 89%           |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset provided by [Florist Customer Churn](https://huggingface.co/datasets/iammkb2002/florist_customer_churn)
- BERT implementation based on [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Citation

If you use Predictix in your research, please cite:

```bibtex
@software{predictix2024,
  title  = {Predictix: Sentiment Analysis for Customer Churn Prediction},
  author = {Amari, Ayudha and Rahadian, Danisa and Erico, Muhammad and Theresa, Yuliana},
  year   = {2024}
}
```
