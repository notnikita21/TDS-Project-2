import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
import requests

# Function to create a folder if it doesn't exist
def create_folder_if_needed(folder_name):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to move files into their respective folders
def move_file_to_folder(file_name, folder_name):
    """Move a file to the specified folder."""
    create_folder_if_needed(folder_name)
    new_path = os.path.join(folder_name, os.path.basename(file_name))
    if os.path.exists(file_name):
        shutil.move(file_name, new_path)
    else:
        print(f"Warning: {file_name} does not exist and cannot be moved.")

# Function to generate and save visualizations for each dataset
def generate_visualizations(data, output_folder, numerical_columns, categorical_columns, datetime_columns):
    # Set figure size limit
    plt.rcParams['figure.figsize'] = (7, 7)

    # Generate pairplot for numerical relationships
    if len(numerical_columns) > 0:
        sns.pairplot(data[numerical_columns])
        plt.savefig(os.path.join(output_folder, 'numerical_relationships.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: No numerical columns available for pairplot.")

    # Generate bar plots for categorical columns
    for col in categorical_columns:
        try:
            if not data[col].empty:
                data[col].value_counts().head(10).plot(kind='bar', title=f"Top Categories in {col}")
                plt.savefig(os.path.join(output_folder, f'{col}_barplot.png'), bbox_inches='tight')
                plt.close()
            else:
                print(f"Warning: Categorical column '{col}' is empty.")
        except Exception as e:
            print(f"Error generating bar plot for {col}: {e}")

    # Generate trend plots for datetime columns
    for col in datetime_columns:
        try:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            if not data[col].isnull().all():  # Check if the column has valid datetime values
                data[col].value_counts().sort_index().plot(title=f"Trends in {col}")
                plt.savefig(os.path.join(output_folder, f'{col}_trend.png'), bbox_inches='tight')
                plt.close()
            else:
                print(f"Warning: Datetime column '{col}' has no valid dates.")
        except Exception as e:
            print(f"Error generating trend plot for {col}: {e}")

# Function to query LLM for insights
def ask_llm(prompt):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

# Main function to process multiple CSV files dynamically
def process_csv_files():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        output_folder = os.path.join(os.getcwd(), dataset_name)
        create_folder_if_needed(output_folder)  # Ensure folder exists

        try:
            data = pd.read_csv(csv_file, encoding='latin-1')
            print(f"Processing {csv_file}...")
            print(f"Columns in {csv_file}: {data.columns.tolist()}")
            print(f"Data types in {csv_file}: {data.dtypes}")

            # Check for required columns (customize as needed)
            required_columns = ['date', 'overall']  # Example required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Error: The following required columns are missing in {csv_file}:
