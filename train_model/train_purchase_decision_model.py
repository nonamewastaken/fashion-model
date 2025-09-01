import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def train_models():
    # Load the dataset
    df = pd.read_csv('research/fashion_trend_data.csv')
    
    # Create a copy for processing
    data = df.copy()
    
    # Handle missing values in Price column
    brand_price = data.groupby("Brand")["Price"].mean()
    
    for index, row in data.iterrows():
        if pd.isnull(row["Price"]):
            data.at[index, "Price"] = brand_price[row["Brand"]]
    
    # Encode categorical variables
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"])
    data["Sentiment"] = le.fit_transform(data["Sentiment"])
    data["Purchase_Decision"] = le.fit_transform(data["Purchase_Decision"])
    
    # Save label encoders
    label_encoders = {
        'Gender': LabelEncoder().fit(df['Gender']),
        'Sentiment': LabelEncoder().fit(df['Sentiment']),
        'Purchase_Decision': LabelEncoder().fit(df['Purchase_Decision'])
    }
    
    # One-hot encoding for Brand and Fashion_Item (without drop_first to keep all columns)
    data = pd.get_dummies(data, columns=["Brand", "Fashion_Item"], drop_first=False)
    
    # Standardize numerical features (including Trendy_Score)
    scaler = StandardScaler()
    data[["Price", "Rating", "Trendy_Score", "Age"]] = scaler.fit_transform(
        data[["Price", "Rating", "Trendy_Score", "Age"]]
    )
    
    # Prepare data for classification model (Purchase Decision)
    # Include Trendy_Score as a feature for purchase decision prediction
    X_class = data.drop(columns=["Purchase_Decision", "Review_Text", "Location", "Review_Date", "User_ID"])
    y_class = data["Purchase_Decision"]
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Train classification model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    # Evaluate classification model
    y_pred_c = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"Classification Accuracy: {acc:.2%}")
    
    # Save models and preprocessing objects
    print("Saving models and preprocessing objects...")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save purchase decision model
    with open('model/purchase_decision_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # Save preprocessing objects
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns for consistency
    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(X_class.columns.tolist(), f)
    
    print("Models saved successfully!")
    print(f"Purchase Decision model accuracy: {acc:.2%}")
    print(f"Feature columns: {X_class.columns.tolist()}")
    
    return clf, label_encoders, scaler, X_class.columns.tolist()

if __name__ == "__main__":
    train_models() 