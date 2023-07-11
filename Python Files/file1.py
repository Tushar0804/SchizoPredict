import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score

df = pd.read_excel("HO5/healthy control1.xlsx")

print("Data Frame Information - df.info()")
print(df.info())

print("\n\n\ndf.describe().T")
print(df.describe().T)

# Label encoding
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

x = df.drop('class', axis=1)
y = df['class']

# Splitting the Dataset for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

# Scaling Of Data 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Dimensionality Reduction - Principle Component Analysis (PCA) 
pca = PCA(n_components=20)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Light Gradient Boosted Machine Model 

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

# Training the model
model = lgb.train(params, train_data, num_boost_round=200)

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Plotting Feature Importance
feature_value = model.feature_importance()
feature_imp = pd.DataFrame(zip(feature_value,x.columns), columns=['Value','Feature'])

plt.figure(figsize=(200, 80))
sb.barplot(x="Value", y="Feature", data=feature_imp)
sb.set(font_scale=15)
plt.show()
