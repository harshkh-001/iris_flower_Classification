import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, encode_species
from src.model import train_model
from src.evaluate import evaluate_model

df = load_data(r"data/IRIS.csv")
df, le = encode_species(df)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)
os.makedirs("results", exist_ok=True)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = df['species_encoded']
evaluate_model(model, X_test, y_test, le, X, Y)
