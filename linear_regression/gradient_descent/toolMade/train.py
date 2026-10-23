import sys
import numpy as np
import pandas as pd

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
        required_columns = ["km", "price"]
        data = open_csv(args[0], required_columns)
    except Exception as e:
        print(f"Error opening CSV file: {e}")
        return

    # Teste usando polyfit
    x_values = data["km"].to_numpy(dtype=float)
    y_values = data["price"].to_numpy(dtype=float)
    t1, t0 = np.polyfit(x_values, y_values, 1)

    print(f"Polyfit parameters: T0 = {t0}, T1 = {t1}")

if __name__ == "__main__":
    main(sys.argv[1:])
