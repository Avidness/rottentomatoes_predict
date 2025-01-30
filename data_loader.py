import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    movies_df = pd.read_csv("data/movies.csv")
    persons_df = pd.read_csv("data/persons.csv")

    movies_df["person_ids"] = movies_df["person_ids"].apply(literal_eval)
    movies_expanded = movies_df.explode("person_ids")
    movies_expanded["person_ids"] = movies_expanded["person_ids"].astype(int)

    # Merge
    movies_expanded = movies_expanded.merge(
        persons_df, left_on="person_ids", right_on="person_id", how="left")

    movies_expanded.drop(columns=["person_id"], inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    movies_expanded["role"] = le.fit_transform(movies_expanded["role"])
    movies_expanded["person_name"] = le.fit_transform(movies_expanded["person_name"])

    # Feature selection
    features = movies_expanded.drop(columns=["movie_id", "movie_title", "audience_rank", "critics_rank"])
    labels = movies_expanded[["audience_rank", "critics_rank"]]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
