from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def train_and_predict(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn_model.fit(X_train_scaled, y_train)
    return nn_model.predict(X_test_scaled)

