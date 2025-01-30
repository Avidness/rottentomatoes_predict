from sklearn.linear_model import LinearRegression

def train_and_predict(X_train, y_train, X_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model.predict(X_test)

