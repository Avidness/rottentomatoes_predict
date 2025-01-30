import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ast import literal_eval

movies_df = pd.read_csv("movies.csv")
persons_df = pd.read_csv("persons.csv")

movies_df["person_ids"] = movies_df["person_ids"].apply(literal_eval)
movies_expanded = movies_df.explode("person_ids")
movies_expanded["person_ids"] = movies_expanded["person_ids"].astype(int)

# Merge
movies_expanded = movies_expanded.merge(
    persons_df, left_on="person_ids", right_on="person_id", how="left")

movies_expanded.drop(columns=["person_id"], inplace=True)

# Encode categorical variables
movies_expanded["role"] = movies_expanded["role"].astype("category").cat.codes
movies_expanded["person_name"] = movies_expanded["person_name"].astype("category").cat.codes

# Feature selection
features = movies_expanded.drop(columns=["movie_id", "movie_title", "audience_rank", "critics_rank"])
labels = movies_expanded[["audience_rank", "critics_rank"]]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)

# Eval
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
