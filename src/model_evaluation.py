import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os

def load_model(model_path):
    """Load the saved model."""
    return joblib.load(model_path)

def load_test_data(file_path):
    """Load test data for evaluation."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Load model
    model_path = os.path.join('models', 'food_nutrition_model.pkl')
    model = load_model(model_path)
    
    # Load test data
    test_data_path = os.path.join('data', 'processed', 'cleaned_food_data.csv')
    df = load_test_data(test_data_path)
    
    # Feature selection (replace with actual feature columns)
    X_test = df[['Mean', 'Median']]
    y_test = df['Group']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print(classification_report(y_test, y_pred))
