import xgboost as xgb

def train_and_predict(X_train, y_train, X_test):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_model.fit(X_train, y_train)
    return xgb_model.predict(X_test)

