import pandas as pd
import numpy as np
import json, sys
from datetime import datetime
from dataclasses import dataclass, field

# Represent a linear regression model with parameters theta0 and theta1
@dataclass
class Model:
    params: np.ndarray = field(default_factory=lambda: [0.0, 0.0])
    scale: float = 100_000

    # Predict the output for a given input x using the linear regression model
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.params  # Using matrix multiplication for prediction


    def save_model(self, file_path: str, training: dict = None) -> None:
        try:
            create_json(
                file_path,
                model={
                    "type": "linear_regression",
                    "parameters": {
                        "T0": self.params[0],
                        "T1": self.params[1] / self.scale
                    },
                    "function": "F(km) = T0 + (T1 * km)",
                },
                training=training if training else None
            )
        except Exception as e:
            raise Exception(f"Error saving model: {e}")


@dataclass
class Trainer:
    data: pd.DataFrame
    model: Model
    scale: float = 100_000
    total_epochs: int = 0
    learning_rate: float = 0.01
    epochs_per_round: int = 1_000
    tolerance: float = 1e-6
    loss: float = float('inf')

    def calculate_loss(self) -> float:
        x = self.data["km"].to_numpy(dtype=float) / self.scale

        x_values = np.column_stack([
            np.ones(len(x)),  # Add a column of ones for the bias term
            x
        ])
        y_values = self.data["price"].to_numpy(dtype=float)

        predictions = self.model.predict(x_values)
        diff = predictions - y_values

        self.loss = (diff ** 2).mean()
        return self.loss


    def update_model(self, gradient: np.ndarray) -> None:
        self.model.params -= gradient


    def train_resume(self) -> dict:
        training_info = {
            "total_epochs": self.total_epochs,
            "learning_rate": self.learning_rate,
            "epochs_per_round": self.epochs_per_round,
            "tolerance": self.tolerance,
            "MSE": self.loss,
            "RMSE": np.sqrt(self.loss)
        }
        return training_info


    def calculate_gradient(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
        predictions = self.model.predict(x_values)
        diff = predictions - y_values

        gradient = (x_values.T @ diff) / len(x_values)

        return self.learning_rate * gradient

    def gradient_descent(self) -> None:
        # Data preparation
        x = self.data["km"].to_numpy(dtype=float) / self.scale
        # x_values is a matrix [[1, x1],
        #                        [1, x2],
        #                        ...,
        x_values = np.column_stack([
            np.ones(len(x)), # Add a column of ones for the bias term
            x
        ])
        y_values = self.data["price"].to_numpy(dtype=float)

        while True:
            self.total_epochs += self.epochs_per_round

            # Predictions and loss calculation
            for _ in range(self.epochs_per_round):
                gradient = self.calculate_gradient(x_values, y_values)
                self.update_model(gradient)

            # Check for convergence
            improvement = abs(self.loss - self.calculate_loss())
            if improvement < self.tolerance:
                print(f"Converged after {self.total_epochs} epochs with loss: {self.loss:.6f}")
                print(f"Final parameters: T0 = {self.model.params[0]}, T1 = {self.model.params[1] / self.scale}")
                break


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
    metadata = {
        **kwargs,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    name = file_path if file_path.endswith('.json') else f"{file_path}.json"

    try:
        with open(name, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
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

    # Train the model
    model = Model()
    trainer = Trainer(data=data, model=model)
    trainer.gradient_descent()
    model.save_model("checkpoint.json", training=trainer.train_resume())

    # Teste usando polyfit
    x_values = data["km"].to_numpy(dtype=float)
    y_values = data["price"].to_numpy(dtype=float)
    t1, t0 = np.polyfit(x_values, y_values, 1)

    print(f"Polyfit parameters: T0 = {t0}, T1 = {t1}")

if __name__ == "__main__":
    main(sys.argv[1:])
