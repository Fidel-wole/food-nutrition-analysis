import pandas as pd
import os

def load_data(file_path):
    """Load the CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handle missing values, outliers, and data types."""
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Print column names to debug
    print("Columns in DataFrame:", df.columns)
    
    # Use the correct column name
    column_to_convert = 'Mean'  

    if column_to_convert in df.columns:
        # Convert the column to integer
        df[column_to_convert] = df[column_to_convert].astype('int')
    else:
        print(f"Warning: '{column_to_convert}' not found in DataFrame")

    return df

def save_processed_data(df, save_path):
    """Save the processed data to the processed folder."""
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Combined_FOOD_METADATA.csv')
    df = load_data(raw_data_path)
    
    # Print the first few rows and columns of the DataFrame
    print("DataFrame head:\n", df.head())
    print("Columns in DataFrame:", df.columns)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Save the cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_food_data.csv')
    save_processed_data(df_cleaned, processed_data_path)
