import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os
import requests

# Use the AIPROXY_TOKEN from the environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set. Please set it as an environment variable.")

# Assuming CSV file is in the repository, or provided via a path argument
dataset_path = 'goodreads.csv'  # Replace with the actual file path or input path

# Read the CSV file with encoding handling
try:
    data = pd.read_csv(dataset_path, encoding='latin-1')  # Adjust encoding if needed
    print("Columns in the dataset and their data types:")
    print(data.dtypes)

    # Show the first 5 rows of the first 20 columns
    print(data.iloc[:, :20].head())

except FileNotFoundError:
    print(f"Error: File '{dataset_path}' not found.")
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}. Try changing the file encoding.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Proceed only if data was read successfully
if 'data' in locals():
    # Summarize the data
    summary = data.describe(include='all')
    missing_values = data.isnull().sum()

    # Extract only numerical columns for correlation matrix
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    if len(numerical_columns) > 0:
        correlation_matrix = data[numerical_columns].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)
    else:
        print("No numerical columns found for correlation matrix.")

    # Data type separation
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    datetime_columns = data.select_dtypes(include=['datetime64']).columns

    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)
    print("Datetime columns:", datetime_columns)

    # Visualizations
    if len(numerical_columns) > 0:
        sns.pairplot(data[numerical_columns])
        plt.savefig("numerical_relationships.png")
        plt.show()
    else:
        print("No numerical columns found.")

    for col in categorical_columns:
        data[col].value_counts().head(10).plot(kind='bar', title=f"Top Categories in {col}")
        plt.savefig(f"{col}_barplot.png")
        plt.show()

    for col in datetime_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        data[col].value_counts().sort_index().plot(title=f"Trends in {col}")
        plt.savefig(f"{col}_trend.png")
        plt.show()

def ask_llm(prompt):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}"
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

# Example usage
prompt = f"Summarize the following dataset insights:\n\n{summary.to_markdown()}"
insights = ask_llm(prompt)
print(insights)

with open('README.md', 'w') as f:
    f.write("# Automated Data Analysis\n\n")
    f.write("## Summary Statistics\n\n")
    f.write(summary.to_markdown())
    f.write("\n\n## Insights\n\n")
    f.write(insights)
    f.write("\n\n## Visualizations\n\n")
    f.write("![Correlation Heatmap](heatmap.png)")

print("Analysis completed. Check the generated README.md.")
