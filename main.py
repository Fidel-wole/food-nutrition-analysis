from src.data_preprocessing import load_data, clean_data, save_processed_data
from src.model_training import train_model, save_model
from src.model_evaluation import load_model, load_test_data
import os
import pandas as pd

def main():
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Combined_FOOD_METADATA.csv')
    df = load_data(raw_data_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_food_data.csv')
    save_processed_data(df_cleaned, processed_data_path)
    
    # Load the cleaned data for training
    df_cleaned = pd.read_csv(processed_data_path)
    
    # Feature selection
    X = df_cleaned[['Mean', 'Median']]
    y = df_cleaned['Group']
    
    # Train the model
    model = train_model(X, y)
    
    # Save the model
    model_save_path = os.path.join('models', 'food_nutrition_model.pkl')
    save_model(model, model_save_path)
    
    # Load the model for evaluation
    model = load_model(model_save_path)
    
    # Test and evaluate the model
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()
