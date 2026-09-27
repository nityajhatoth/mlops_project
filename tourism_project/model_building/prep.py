
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} duplicate row(s).")
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    return df

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Xtrain shape          : {Xtrain.shape}")
    print(f"Xtest shape           : {Xtest.shape}")
    print(f"ytrain shape          : {ytrain.shape}")
    print(f"ytest shape           : {ytest.shape}")

if __name__ == "__main__":
    prepare_data()
