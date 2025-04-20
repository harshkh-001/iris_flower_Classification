import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def encode_species(df):
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])
    return df, le