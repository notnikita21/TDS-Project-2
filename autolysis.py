import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import shutil
import uvicorn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.model_selection import train_test_split

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
    shutil.move(file_name, new_path)

# Function to generate and save visualizations for each dataset
def generate_visualizations(data, output_folder, numerical_columns, categorical_columns, datetime_columns):
    """Generate and save visualizations."""

    # Set figure size limit
    plt.rcParams['figure.figsize'] = (7, 7)

    # Outlier and Anomaly Detection: Boxplot for numerical columns
    for col in numerical_columns:
        sns.boxplot(data[col])
        plt.title(f"Outlier Detection in {col}")
        plt.savefig(os.path.join(output_folder, f'{col}_outliers.png'), bbox_inches='tight')
        plt.close()

    # Correlation Heatmap for numerical relationships
    if len(numerical_columns) > 0:
        correlation_matrix = data[numerical_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Analysis")
        plt.savefig(os.path.join(output_folder, 'correlation_analysis.png'), bbox_inches='tight')
        plt.close()

    # Generate pairplot for numerical relationships
    if len(numerical_columns) > 0:
        sns.pairplot(data[numerical_columns])
        plt.savefig(os.path.join(output_folder, 'numerical_relationships.png'), bbox_inches='tight')
        plt.close()

    # Regression Analysis: Linear regression plot
    if len(numerical_columns) > 1:
        X = data[numerical_columns].dropna().iloc[:, :-1]  # Features (excluding the last column)
        y = data[numerical_columns].dropna().iloc[:, -1]  # Target (last column)
        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)

        plt.scatter(y, y_pred)
        plt.title("Regression Analysis: Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.savefig(os.path.join(output_folder, 'regression_analysis.png'), bbox_inches='tight')
        plt.close()

    # Feature Importance Analysis using Random Forest
    if len(numerical_columns) > 1:
        X = data[numerical_columns].dropna().iloc[:, :-1]
        y = data[numerical_columns].dropna().iloc[:, -1]

        rf = RandomForestRegressor()
        rf.fit(X, y)
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

        feature_importance.sort_values(ascending=False).plot(kind='bar', title="Feature Importance")
        plt.savefig(os.path.join(output_folder, 'feature_importance.png'), bbox_inches='tight')
        plt.close()

    # Cluster Analysis: KMeans clustering
    if len(numerical_columns) > 1:
        X = data[numerical_columns].dropna()
        kmeans = KMeans(n_clusters=4, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X)
        sns.scatterplot(data=data, x=numerical_columns[0], y=numerical_columns[1], hue='Cluster', palette='Set2')
        plt.title("Cluster Analysis")
        plt.savefig(os.path.join(output_folder, 'cluster_analysis.png'), bbox_inches='tight')
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
    """Query the LLM for insights."""
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

# Main function to process a single CSV file dynamically
def process_csv_file(csv_file):
    """Process the given CSV file and generate analysis and visualizations."""
    dataset_name = os.path.splitext(csv_file)[0]
    output_folder = os.path.join(os.getcwd(), dataset_name)
    create_folder_if_needed(output_folder)  # Ensure folder exists

    try:
        data = pd.read_csv(csv_file, encoding='latin-1')
        print(f"Processing {csv_file}...")

        # Get column details
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        datetime_columns = data.select_dtypes(include=['datetime64']).columns

        summary = data.describe(include='all')

        # Generate visualizations
        generate_visualizations(data, output_folder, numerical_columns, categorical_columns, datetime_columns)

        # Ask LLM for insights
        prompt = f"Summarize the following dataset insights:\n\n{summary.to_markdown()}"
        insights = ask_llm(prompt)

        # Write the README file
        readme_path = os.path.join(output_folder, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"# Automated Data Analysis for {os.path.basename(csv_file)}\n\n")
            f.write("## Summary Statistics\n\n")
            f.write(summary.to_markdown())
            f.write("\n\n## Insights\n\n")
            f.write(insights)
            f.write("\n\n## Visualizations\n\n")

            # Add visualizations to the README
            f.write(f"![Correlation Analysis](correlation_analysis.png)\n")
            if len(numerical_columns) > 0:
                f.write(f"![Numerical Relationships](numerical_relationships.png)\n")
            for col in categorical_columns:
                f.write(f"![{col} Bar Plot]({col}_barplot.png)\n")
            for col in datetime_columns:
                f.write(f"![{col} Trends]({col}_trend.png)\n")
            f.write(f"![Outlier Detection]({numerical_columns[0]}_outliers.png)\n")
            f.write(f"![Regression Analysis](regression_analysis.png)\n")
            f.write(f"![Feature Importance](feature_importance.png)\n")
            f.write(f"![Cluster Analysis](cluster_analysis.png)\n")

        # Move the generated README and visualizations to the folder
        move_file_to_folder('README.md', output_folder)
        for col in categorical_columns:
            move_file_to_folder(f'{col}_barplot.png', output_folder)
        for col in datetime_columns:
            move_file_to_folder(f'{col}_trend.png', output_folder)
        move_file_to_folder('numerical_relationships.png', output_folder)
        move_file_to_folder('correlation_analysis.png', output_folder)
        move_file_to_folder('regression_analysis.png', output_folder)
        move_file_to_folder('feature_importance.png', output_folder)
        move_file_to_folder('cluster_analysis.png', output_folder)

        # Also move the CSV file into the folder
        move_file_to_folder(csv_file, output_folder)

        print(f"Analysis completed for {csv_file}. Check the generated folder '{output_folder}'.")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

# Entry point when script is executed
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uvicorn autolysis.py dataset.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    process_csv_file(csv_file)
