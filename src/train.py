import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def train_churn_model():
    # 1. Load the Training Data
    # Path relative to your project root
    try:
        df = pd.read_csv("data/raw/customer_churn_dataset-training-master.csv")
    except FileNotFoundError:
        print("Error: CSV file not found in data/raw/")
        return
    
    # 2. Cleaning & Initial Preprocessing
    # Dropping CustomerID (noise) and handling missing values
    df = df.drop(['CustomerID'], axis=1).dropna()
    
    # 3. Encoding Categorical Features
    # Converting text columns into numbers for the model
    le = LabelEncoder()
    cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 4. Feature Selection
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 5. Professional Data Split
    # Using stratify=y to maintain churn balance in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Model Training
    print("🚀 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    # 7. Evaluation Report
    y_pred = model.predict(X_val)
    print("\nModel Performance Report:")
    print(classification_report(y_val, y_pred))
    
    # 8. Save Model Artifact
    joblib.dump(model, "models/churn_model_v1.joblib")
    print("\n✅ Success! Model saved in models/churn_model_v1.joblib")

if __name__ == "__main__":
    train_churn_model()