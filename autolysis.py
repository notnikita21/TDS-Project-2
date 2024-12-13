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

# Get dataset path dynamically
dataset_path = input("Enter the CSV file path: ").strip()
output_folder = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file with encoding handling
try:
    data = pd.read_csv(dataset_path, encoding='latin-1')
    print("Columns in the dataset and their data types:")
    print(data.dtypes)
    print(data.iloc[:, :20].head())
except FileNotFoundError:
    raise FileNotFoundError(f"Error: File '{dataset_path}' not found.")
except pd.errors.EmptyDataError:
    raise ValueError("Error: The CSV file is empty.")
except UnicodeDecodeError as e:
    raise ValueError(f"Encoding error: {e}. Try changing the file encoding.")

# Summarize the data
summary = data.describe(include='all')
missing_values = data.isnull().sum()

# Extract numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
if len(numerical_columns) > 0:
    correlation_matrix = data[numerical_columns].corr()
else:
    print("No numerical columns found for correlation matrix.")

# Data type separation
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
datetime_columns = data.select_dtypes(include=['datetime64']).columns

# Visualizations
if len(numerical_columns) > 0:
    sns.pairplot(data[numerical_columns])
    plt.savefig(f"{output_folder}/numerical_relationships.png")
    plt.close()

for col in categorical_columns:
    data[col].value_counts().head(10).plot(kind='bar', title=f"Top Categories in {col}")
    plt.savefig(f"{output_folder}/{col}_barplot.png")
    plt.close()

for col in datetime_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    data[col].value_counts().sort_index().plot(title=f"Trends in {col}")
    plt.savefig(f"{output_folder}/{col}_trend.png")
    plt.close()

# Function to query LLM
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

# Generate insights
prompt = f"Summarize the following dataset insights:\n\n{summary.to_markdown()}"
insights = ask_llm(prompt)
print(insights)

# Write README.md
readme_path = os.path.join(output_folder, 'README.md')
with open(readme_path, 'w') as f:
    f.write("# Automated Data Analysis\n\n")
    f.write("## Summary Statistics\n\n")
    f.write(summary.to_markdown())
    f.write("\n\n## Insights\n\n")
    f.write(insights)
    f.write("\n\n## Visualizations\n\n")
    if len(numerical_columns) > 0:
        f.write(f"![Numerical Relationships](numerical_relationships.png)\n")
    for col in categorical_columns:
        f.write(f"![{col} Bar Plot]({col}_barplot.png)\n")
    for col in datetime_columns:
        f.write(f"![{col} Trends]({col}_trend.png)\n")

print(f"Analysis completed. Check the generated folder '{output_folder}'.")
