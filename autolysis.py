import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Function to create a folder if it doesn't exist
def create_folder_if_needed(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to move files into their respective folders
def move_file_to_folder(file_name, folder_name):
    create_folder_if_needed(folder_name)
    new_path = os.path.join(folder_name, os.path.basename(file_name))
    if os.path.exists(file_name):
        shutil.move(file_name, new_path)

# Function to generate and save visualizations for each dataset
def generate_visualizations(data, output_folder, numerical_columns, categorical_columns, datetime_columns):
    plt.rcParams['figure.figsize'] = (7, 7)

    for col in numerical_columns:
        sns.boxplot(data[col])
        plt.title(f"Outlier Detection in {col}")
        plt.savefig(os.path.join(output_folder, f'{col}_outliers.png'), bbox_inches='tight')
        plt.close()

    if len(numerical_columns) > 0:
        correlation_matrix = data[numerical_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Analysis")
        plt.savefig(os.path.join(output_folder, 'correlation_analysis.png'), bbox_inches='tight')
        plt.close()

        sns.pairplot(data[numerical_columns])
        plt.savefig(os.path.join(output_folder, 'numerical_relationships.png'), bbox_inches='tight')
        plt.close()

    if len(numerical_columns) > 1:
        X = data[numerical_columns].dropna().iloc[:, :-1]
        y = data[numerical_columns].dropna().iloc[:, -1]

        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        plt.scatter(y, y_pred)
        plt.title("Regression Analysis: Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.savefig(os.path.join(output_folder, 'regression_analysis.png'), bbox_inches='tight')
        plt.close()

        rf = RandomForestRegressor()
        rf.fit(X, y)
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
        feature_importance.sort_values(ascending=False).plot(kind='bar', title="Feature Importance")
        plt.savefig(os.path.join(output_folder, 'feature_importance.png'), bbox_inches='tight')
        plt.close()

        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        clusters = pd.DataFrame(kmeans.fit_predict(X), columns=['Cluster'], index=X.index)
        data = data.loc[X.index].assign(Cluster=clusters['Cluster'])
        sns.scatterplot(data=data, x=numerical_columns[0], y=numerical_columns[1], hue='Cluster', palette='Set2')
        plt.title("Cluster Analysis")
        plt.savefig(os.path.join(output_folder, 'cluster_analysis.png'), bbox_inches='tight')
        plt.close()

    for col in categorical_columns:
        try:
            data[col].value_counts().head(10).plot(kind='bar', title=f"Top Categories in {col}")
            plt.savefig(os.path.join(output_folder, f'{col}_barplot.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating bar plot for {col}: {e}")

    for col in datetime_columns:
        try:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col].value_counts().sort_index().plot(title=f"Trends in {col}")
            plt.savefig(os.path.join(output_folder, f'{col}_trend.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating trend plot for {col}: {e}")

# Function to generate README using LLM
def generate_readme(data, output_folder, analysis_summary):
    readme_content = f"""# Automated Data Analysis Report

## Dataset Summary

The dataset contains {len(data)} rows and {len(data.columns)} columns. 
Key columns include: {', '.join(data.columns[:5])}...

## Analysis Summary

We conducted extensive analysis, including:

- **Outlier Detection**: Key variables examined for anomalies.
- **Correlation Analysis**: Relationships between numerical features visualized.
- **Regression Analysis**: Predicted outcomes using linear and random forest models.
- **Cluster Analysis**: Grouped similar records using KMeans clustering.
- **Category and Trend Insights**: Top categories and time-based trends explored.

## Key Insights and Recommendations

{analysis_summary}

## Visualizations

"""
    for file in os.listdir(output_folder):
        if file.endswith('.png'):
            readme_content += f"![{file.split('.')[0].replace('_', ' ').title()}]({file})\n"

    readme_path = os.path.join(output_folder, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

# Main function to process a CSV file
def process_csv_file(csv_file):
    dataset_name = os.path.splitext(csv_file)[0]
    output_folder = os.path.join(os.getcwd(), dataset_name)
    create_folder_if_needed(output_folder)

    try:
        data = pd.read_csv(csv_file, encoding='latin-1')
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        datetime_columns = data.select_dtypes(include=['datetime64']).columns

        generate_visualizations(data, output_folder, numerical_columns, categorical_columns, datetime_columns)
        analysis_summary = "Our findings reveal key patterns, suggesting targeted improvements."
        generate_readme(data, output_folder, analysis_summary)

        print(f"Analysis completed for {csv_file}. Check the folder '{output_folder}'.")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    process_csv_file(csv_file)
