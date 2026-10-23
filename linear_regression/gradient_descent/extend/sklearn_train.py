from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import sys

def open_csv(
    file_path: str,
    required_columns: list
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df[required_columns]
    except Exception as e:
        raise Exception(e)


def main(args):
    if len(args) == 0:
        print("Usage: python train.py <data_file.csv>")
        sys.exit(1)

    # Load data
    try:
        required_columns = ["km", "engine_size", "year", "brand", "model", "price"]
        data = open_csv(args[0], required_columns)
    except Exception as e:
        print(f"Error opening CSV file: {e}")
        return

    # Prepare features and target variable
    price = data["price"].to_numpy(dtype=float)
    km = data["km"].to_numpy(dtype=float)
    year = data["year"].to_numpy(dtype=float)
    engine_size = data["engine_size"].to_numpy(dtype=float)
    brand = pd.get_dummies(data["brand"], drop_first=True).to_numpy(dtype=float)
    model = pd.get_dummies(data["model"], drop_first=True).to_numpy(dtype=float)

    # normalization
    km = (km - km.mean()) / km.std()
    year = (year - year.mean()) / year.std()
    engine_size = (engine_size - engine_size.mean()) / engine_size.std()

    x_values = np.column_stack([
        km,
        year,
        engine_size,
        brand,
        model
    ])
    y_values = price

    # Train the model
    model = LinearRegression()
    model.fit(x_values, y_values)

    print("Model trained successfully.")
    print(f"Coefficients: {model.coef_}")
    print("MSE: ", np.mean((model.predict(x_values) - y_values) ** 2))
    print("RMSE: ", np.sqrt(np.mean((model.predict(x_values) - y_values) ** 2)))

    
if __name__ == "__main__":
   main(sys.argv[1:])