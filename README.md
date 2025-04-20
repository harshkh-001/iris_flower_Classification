# Iris Flower Classification

This project involves building a machine learning model to classify the species of Iris flowers based on their physical features using the well-known Iris dataset. The project includes data loading, preprocessing, model training, evaluation, and visualization.

## Project Structure

The project is organized as follows:

iris-flower-classification/ │ ├── data/ # Raw dataset │ └── iris.csv │ ├── notebooks/ # Jupyter notebooks for EDA & experiments │ └── eda_and_model.ipynb │ ├── src/ # Modular codebase │ ├── preprocessing.py # Data loading and preprocessing │ ├── model.py # Model training │ └── evaluate.py # Evaluation and visualization │ ├── results/ # Output results │ ├── metrics.txt # Model evaluation metrics │ └── confusion_matrix.png # Confusion matrix visualization │ ├── README.md # Documentation for the project ├── requirements.txt # Python dependencies └── main.py


## Requirements

The project requires the following Python packages:

- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

You can install the required dependencies using the following command:


## Usage

### Step 1: Prepare Your Environment

1. Clone the repository or download the files to your local machine.
2. Install the required dependencies using the following command:


### Step 2: Run the Project

To run the project, execute the following command:


This will:

1. Load the Iris dataset from `data/iris.csv`.
2. Preprocess the data (encoding the target variable).
3. Train a Random Forest model.
4. Evaluate the model and save the metrics to `results/metrics.txt`.
5. Save the confusion matrix plot to `results/confusion_matrix.png`.

### Step 3: Explore the Notebooks

The Jupyter notebook located at `notebooks/data_checking.ipynb` contains the following:

- Exploratory Data Analysis (EDA) using Pandas and Seaborn.
- Visualizations like pair plots and correlation matrices.

## File Descriptions

- **data/iris.csv**: The Iris dataset, which contains 150 rows and 5 columns: sepal length, sepal width, petal length, petal width, and species.
- **notebooks/eda_and_model.ipynb**: Jupyter notebook for exploratory data analysis (EDA) and initial experiments.
- **src/preprocessing.py**: Code for loading and preprocessing the data, including label encoding of the target variable.
- **src/model.py**: Code to define and train the machine learning model (Random Forest).
- **src/evaluate.py**: Code for evaluating the model, generating the classification report, and saving the confusion matrix.
- **results/metrics.txt**: Model evaluation metrics in text format.
- **results/confusion_matrix.png**: Confusion matrix plot saved as an image.
- **main.py**: Orchestrates the entire process, from loading data to training the model and saving the results.

