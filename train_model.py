import os, joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def train_and_export():
    dataset_path = os.path.join("data", "properties_1000.xlsx")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")

    df = pd.read_excel(dataset_path)

    df_encoded = pd.get_dummies(df, columns=['Locality', 'Property_Type'], drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c not in ['Property_ID', 'Price_Lakhs']]
    X = df_encoded[feature_cols]
    y = df_encoded['Price_Lakhs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"Model Evaluation -> R² Score: {r2:.4f} | RMSE: {rmse:.2f} Lakhs | MAPE: {mape*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, os.path.join("models", "prop_val_xgb.pkl"))
    print("Serialized model artifact to models/prop_val_xgb.pkl")

if __name__ == "__main__":
    train_and_export()
