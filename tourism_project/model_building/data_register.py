import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS = ["CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier","DurationOfPitch",
                    "Occupation", "Gender", "NumberOfPersonVisiting","NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
                    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
                    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",]

def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    assert os.path.exists(path), (
        f"Dataset not found at {path}. Make sure tourism.csv has been "
        "committed to the repository's data folder.")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing_columns, f"Missing expected columns: {missing_columns}"
    print("Dataset registered successfully.")
    print(f"Path      : {path}")
    print(f"Shape     : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns   : {list(df.columns)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts())
    print(df["ProdTaken"].value_counts(normalize=True).rename("proportion"))
    return df

if __name__ == "__main__":
    register_dataset()
