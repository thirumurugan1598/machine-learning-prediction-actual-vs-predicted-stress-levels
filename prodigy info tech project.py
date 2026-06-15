import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the Dataset
file_name = "C:/Users/sarve/Downloads/archive/student-lifestyle-and-stress-dataset.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Ensure that '{file_name}' is located in your execution directory.")

df = pd.read_csv(file_name)

# 2. Select Features and Target Variable
features = [
    "Sleep_Hours",
    "Study_Hours",
    "Social_Media_Hours",
    "Attendance",
    "Exam_Pressure",
    "Family_Support",
]
target = "Stress_Level"

# Clean rows with missing values
cleaned_data = df[features + [target]].dropna()

X = cleaned_data[features]
y = cleaned_data[target]

# 3. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("=== Linear Regression Model Parameters ===")
print(f"Intercept (beta_0): {model.intercept_:.4f}")
for feature_name, coef_val in zip(features, model.coef_):
    print(f"Coefficient for {feature_name} (beta): {coef_val:.4f}")

print("\n=== Performance Metrics ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R2): {r2:.4f}")

# 6. Generate Plot and Save
plt.scatter(y_test, y_pred, alpha=0.05, color="purple")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    lw=2,
    linestyle="--",
)
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Actual vs Predicted Stress Levels")
plt.tight_layout()
plt.savefig("actual_vs_predicted_stress.png")
plt.show()
