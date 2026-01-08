

# 🚀 End-to-End Customer Churn Prediction System



This project implements a production-ready machine learning pipeline to predict customer churn. It covers the full lifecycle from Exploratory Data Analysis (EDA) to a containerized API deployment.



## 📊 Key Insights from EDA

\* \*\*Support Calls:\*\* A critical churn driver; customers with >5 calls show a significantly higher churn rate.

\* \*\*Payment Delays:\*\* Financial friction (delays >15 days) is strongly correlated with account cancellation.

\* \*\*Total Spend:\*\* Lower-to-mid tier spenders are currently the most at-risk demographic.



## 🛠️ Tech Stack

\* \*\*Machine Learning:\*\* Scikit-Learn (Random Forest Classifier)

\* \*\*API Framework:\*\* FastAPI

\* \*\*Containerization:\*\* Docker

\* \*\*Data Science:\*\* Pandas, NumPy, Seaborn, Matplotlib



## 🏗️ Project Structure

\- `src/train.py`: Production script for model training and artifact generation.

\- `app.py`: FastAPI application serving real-time predictions.

\- `notebooks/`: Experimental discovery and data visualization.

\- `models/`: Versioned model artifacts (.joblib).

\- `Dockerfile`: Configuration for containerized deployment.



## 🚀 Getting Started

1\. \*\*Train the Model:\*\* `python src/train.py`

2\. \*\*Launch the API:\*\* `uvicorn app:app --reload`

3\. \*\*Containerize:\*\* `docker build -t churn-predictor .`



