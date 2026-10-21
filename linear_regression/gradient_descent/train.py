import pandas as pd
import numpy as np
import json, sys
from datetime import datetime
from dataclasses import dataclass

# Represent a linear regression model with parameters theta0 and theta1
@dataclass
class Model:
    theta0: float = 0
    theta1: float = 0
    scale: float = 100_000

    # Predict the output for a given input x using the linear regression model
    def predict(self, x: float) -> float:
        return self.theta0 + (self.theta1 * x)
    

    def save_model(self, file_path: str, training: dict = None) -> None:
        try:
            create_json(
                file_path,
                model={
                    "type": "linear_regression",
                    "parameters": {
                        "T0": self.theta0,
                        "T1": self.theta1 / self.scale
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
        x_values = self.data["km"].to_numpy(dtype=float) / self.scale
        y_values = self.data["price"].to_numpy(dtype=float)

        predictions = self.model.predict(x_values)
        diff = predictions - y_values

        self.loss = (diff ** 2).mean()
        return self.loss


    def update_model(self, tmp_theta0: float, tmp_theta1: float) -> None:
        self.model.theta0 -= tmp_theta0
        self.model.theta1 -= tmp_theta1


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

        tmp_theta0 = self.learning_rate * diff.mean()
        tmp_theta1 = self.learning_rate * (diff * x_values).mean()

        return tmp_theta0, tmp_theta1

    def gradient_descent(self) -> None:
        # Data preparation
        x_values = self.data["km"].to_numpy(dtype=float) / self.scale
        y_values = self.data["price"].to_numpy(dtype=float)

        while True:
            self.total_epochs += self.epochs_per_round

            # Predictions and loss calculation
            for _ in range(self.epochs_per_round):
                tmp_theta0, tmp_theta1 = self.calculate_gradient(x_values, y_values)
                self.update_model(tmp_theta0, tmp_theta1)

            # Check for convergence
            improvement = abs(self.loss - self.calculate_loss())
            if improvement < self.tolerance:
                print(f"Converged after {self.total_epochs} epochs with loss: {self.loss:.6f}")
                print(f"Final parameters: T0 = {self.model.theta0}, T1 = {self.model.theta1 / self.scale}")
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

    # Setup training parameters
    LEANING_RATE = 0.01
    EPOCHS = 1000
    TOLERANCE = 1e-5
    SCALE = 100_000

    # Train the model
    model = Model(theta0=0.0, theta1=0.0, scale=SCALE)
    trainer = Trainer(
        data=data,
        scale=SCALE,
        learning_rate=LEANING_RATE,
        epochs_per_round=EPOCHS,
        tolerance=TOLERANCE,
        model=model,
    )
    trainer.gradient_descent()
    model.save_model("checkpoint.json", training=trainer.train_resume())


if __name__ == "__main__":
    main(sys.argv[1:])
