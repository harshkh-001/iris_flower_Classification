# Iris Flower Classification

This project involves building a machine learning model to classify the species of Iris flowers based on their physical features using the well-known Iris dataset. The project includes data loading, preprocessing, model training, evaluation, and visualization.

## Project Structure

The project is organized as follows:

iris-flower-classification/<br>
│<br>
├── data/<br>
│   └── iris.csv<br>
│<br>
├── notebooks/<br>
│   └── data_checking.ipynb<br>
│<br>
├── src/<br>
│   ├── preprocessing.py<br>
│   ├── model.py<br>
│   └── evaluate.py<br>
│<br>
├── results/<br>
│   ├── metrics.txt<br>
│   ├── confusion_matrix.png<br>
│   └── Feature Importance.png<br>
│<br>
├── README.md<br>
├── requirements.txt<br>
└── main.py<br>


## Requirements

The project requires the following Python packages:

- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

You can install the required dependencies using the following command:

## Data Preprocessing

### 1. Loading Data
The raw dataset, `iris.csv`, is loaded from the `data/` directory. It consists of 149 instances of Iris flowers, each with 4 features: Sepal Length, Sepal Width, Petal Length, and Petal Width. The target variable is the species of the flower, with 3 classes:

- **Setosa**
- **Versicolor**
- **Virginica**

### 2. Handling Missing Data
No Missing Data Found (No value is null)

### 3. Label Encoding
The target variable (`species`) is categorical, so it is label-encoded into numerical values for model training.

## Model Selection

### 1. Random Forest Classifier
We chose the **Random Forest** classifier for this classification task. It is an ensemble learning method that uses multiple decision trees to improve prediction accuracy and prevent overfitting. This model was selected due to its performance on classification tasks and its ability to handle high-dimensional data effectively.

### 2. Hyperparameter Tuning
The hyperparameters of the Random Forest model, such as the number of trees (`n_estimators`), maximum depth (`max_depth`), and minimum samples for split (`min_samples_split`), are optimized using **GridSearchCV** to find the best combination for the dataset.

### 3. Model Training
The model is trained using the preprocessed data. The training process involves fitting the Random Forest classifier to the data and evaluating its performance on the test set.

## Model Evaluation

### 1. Accuracy Score
The model’s **accuracy** is calculated by comparing the predicted values with the actual values from the test set.

### 2. Confusion Matrix
A **confusion matrix** is generated to show how well the model is performing in terms of true positives, false positives, true negatives, and false negatives.

### 3. Feature Importance
The **feature importance** is calculated to identify which features contribute the most to the model's decision-making process. This is visualized in a bar plot.

### 4. Classification Report
A detailed **classification report** is generated, showing precision, recall, F1-score, and support for each class.

### 5. Evaluation Results
The model performance metrics are saved to the `results/metrics.txt` file. A confusion matrix and feature importance plot are saved as PNG images in the `results/` folder.


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
6. Save the feature importance plot to `results/feature_importance.png`.

### Step 3: Explore the Notebooks

The Jupyter notebook located at `notebooks/data_checking.ipynb` contains the following:

- Exploratory Data Analysis (EDA) using Pandas and Seaborn.
- Visualizations like pair plots and correlation matrices.

## File Descriptions

- **data/iris.csv**: The Iris dataset, which contains 150 rows and 5 columns: sepal length, sepal width, petal length, petal width, and species.
- **notebooks/eda_and_model.ipynb**: Jupyter notebook for exploratory data analysis (EDA) and initial experiments.
- **src/preprocessing.py**: Code for loading and preprocessing the data, including label encoding of the target variable.
- **src/model.py**: Code to define and train the machine learning model (Random Forest).
- **src/evaluate.py**: Code for evaluating the model, generating the classification report, and saving the confusion matrix and feature importance bar graph.
- **results/metrics.txt**: Model evaluation metrics in text format.
- **results/confusion_matrix.png**: Confusion matrix plot saved as an image.
- **main.py**: Orchestrates the entire process, from loading data to training the model and saving the results.

