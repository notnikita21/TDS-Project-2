import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import requests

# Function to generate and save visualizations for each dataset
def generate_visualizations(data, output_folder):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    datetime_columns = data.select_dtypes(include=['datetime64']).columns

    # Set figure size limit
    plt.rcParams['figure.figsize'] = (7, 7)

    # Generate pairplot for numerical relationships
    if len(numerical_columns) > 0:
        sns.pairplot(data[numerical_columns])
        plt.savefig(os.path.join(output_folder, 'numerical_relationships.png'), bbox_inches='tight')
        plt.close()

    # Generate bar plots for categorical columns
    for col in categorical_columns:
        try:
            data[col].value_counts().head(10).plot(kind='bar', title=f"Top Categories in {col}")
            plt.savefig(os.path.join(output_folder, f'{col}_barplot.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating bar plot for {col}: {e}")

    # Generate trend plots for datetime columns
    for col in datetime_columns:
        try:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col].value_counts().sort_index().plot(title=f"Trends in {col}")
            plt.savefig(os.path.join(output_folder, f'{col}_trend.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating trend plot for {col}: {e}")

# Function to query LLM for insights
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

# Main function to process multiple CSV files dynamically
def process_csv_files():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        output_folder = os.path.join(os.getcwd(), dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        try:
            data = pd.read_csv(csv_file, encoding='latin-1')
            print(f"Processing {csv_file}...")
            print(f"Columns in {csv_file}: {data.columns.tolist()}")
            print(f"Data types in {csv_file}: {data.dtypes}")

            summary = data.describe(include='all')
            generate_visualizations(data, output_folder)

            prompt = f"Summarize the following dataset insights:\n\n{summary.to_markdown()}"
            insights = ask_llm(prompt)

            readme_path = os.path.join(output_folder, 'README.md')
            with open(readme_path, 'w') as f:
                f.write(f"# Automated Data Analysis for {os.path.basename(csv_file)}\n\n")
                f.write("## Summary Statistics\n\n")
                f.write(summary.to_markdown())
                f.write("\n\n## Insights\n\n")
                f.write(insights)
                f.write("\n\n## Visualizations\n\n")

                if len(data.select_dtypes(include=['float64', 'int64']).columns) > 0:
                    f.write(f"![Numerical Relationships](numerical_relationships.png)\n")
                for col in categorical_columns:
                    f.write(f"![{col} Bar Plot]({col}_barplot.png)\n")
                for col in datetime_columns:
                    f.write(f"![{col} Trends]({col}_trend.png)\n")

            print(f"Analysis completed for {csv_file}. Check the generated folder '{output_folder}'.")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    process_csv_files()
