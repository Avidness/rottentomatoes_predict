from sklearn.ensemble import RandomForestRegressor

def train_and_predict(X_train, y_train, X_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model.predict(X_test)

