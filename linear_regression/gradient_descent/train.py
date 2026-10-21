import pandas as pd
import json,sys
from datetime import datetime

# Utils
def open_csv(
    file_path: str,
    required_columns: list
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df[required_columns]
    except Exception as e:
        raise Exception(e)


def create_json(
    file_path : str,
    **kwargs: dict
):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = kwargs.copy()
    metadata["last_updated"] = date
    name = file_path if file_path.endswith('.json') else f"{file_path}.json"

    try:
        with open(name, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
    except Exception as e:
        raise Exception(e)



def train(
    data: pd.DataFrame,
    learning_rate: float,
    epochs: int,
    init_theta0: float = 0.0,
    init_theta1: float = 0.0,
    SCALE: int = 100_000
) -> tuple[float, float, float]:

    theta0 = init_theta0
    theta1 = init_theta1

    print(f"Training with initial parameters: T0 = {theta0}, T1 = {theta1}")

    x_values = data["km"].to_numpy(dtype=float) / SCALE
    y_values = data["price"].to_numpy(dtype=float)

    for _ in range(epochs):
        predictions = theta0 + (theta1 * x_values)
        diff = predictions - y_values
        loss = (diff ** 2).mean()

        tmp_theta0 = learning_rate * diff.mean()
        tmp_theta1 = learning_rate * (diff * x_values).mean()

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1, loss

def main():
    try:
        required_columns = ["km", "price"]
        data = open_csv("data.csv", required_columns)
    except Exception as e:
        print(f"Error opening CSV file: {e}")
        return

    # Setup
    epochs_per_round = 1_000
    learning_rate = 0.01
    tolerance = 1e-05
    SCALE = 100_000

    # Results
    theta0 = 0.0
    theta1 = 0.0
    loss = float('inf')
    total_epochs = 0

    i = 0
    while True:
        total_epochs += epochs_per_round

        print(f"Starting training round {i + 1} with T0 = {theta0}, T1 = {theta1}, Loss = {loss}")
        i += 1

        # Train the model
        theta0, theta1, actual_loss = train(
            data=data,
            learning_rate=learning_rate,
            epochs=epochs_per_round,
            init_theta0=theta0,
            init_theta1=theta1,
            SCALE=SCALE
        )

        improvement = loss - actual_loss
        loss = actual_loss

        if improvement < tolerance:
            print(f"Training completed in {total_epochs} epochs.")
            print(f"Learned parameters: T0 = {theta0}, T1 = {theta1}")
            print(f"Final loss: {loss}")
            break

    try:
        create_json(
            "model.json",
            model={
                "type": "linear_regression",
                "parameters": {
                    "T0": theta0,
                    "T1": theta1 / SCALE
                },
                "function": "F(km) = T0 + (T1 * km)",
            },
            training={
                "learning_rate": learning_rate,
                "epochs": total_epochs,
                "tolerance": tolerance,
                "loss": loss
            }
        )
    except Exception as e:
        print(f"Error creating JSON: {e}")
        return

if __name__ == "__main__":
    main()