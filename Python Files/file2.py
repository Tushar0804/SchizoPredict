import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

# LABEL ENCODING
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

x = df.drop('Class', axis=1)
x = x.drop('Time', axis=1)
y = df['Class']

# SPLITTING THE DATASET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)

# Creating LightGBM dataset
train_data = lgb.Dataset(x_train, label=y_train)

# Set LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'dart',
    'num_leaves': 200,
    'learning_rate': 0.6,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=200)

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
