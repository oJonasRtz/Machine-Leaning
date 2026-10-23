import sys
import json
import numpy as np


SCALE = 100_000


def get_model(filepath: str) -> tuple[np.ndarray, list[str]]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        model = data.get("model", {})
        training = data.get("training", {})

        parameters = model.get("parameters", {})
        feature_names = training.get("feature_names", [])

        params = np.array([
            float(parameters[key])
            for key in sorted(
                parameters,
                key=lambda key: int(key[1:])
            )
        ])

        return params, feature_names

    except (OSError, json.JSONDecodeError) as error:
        print(f"Error reading model file: {error}")
        return np.array([]), []


def prepare_input(
    km: float,
    year: float,
    engine_size: float,
    brand: str,
    model: str,
    feature_names: list[str]
) -> np.ndarray:

    brand_features = feature_names[3:6]
    model_features = feature_names[6:]

    if brand not in brand_features and brand != "Chevrolet":
        raise ValueError(f"Unknown brand: {brand}")

    if model not in model_features and model != "Celta":
        raise ValueError(f"Unknown model: {model}")

    features = [
        km,
        year,
        engine_size
    ]

    features.extend(
        float(brand == category)
        for category in brand_features
    )

    features.extend(
        float(model == category)
        for category in model_features
    )

    return np.array([1.0] + features)


def main(args):
    if len(args) != 5:
        print(
            "Usage: python service.py "
            "<km> <year> <engine_size> <brand> <model>"
        )
        return

    try:
        km = float(args[0])
        year = float(args[1])
        engine_size = float(args[2])
        brand = args[3]
        model = args[4]

        parameters, feature_names = get_model("checkpoint.json")

        if len(parameters) == 0:
            print("Error: no model parameters found.")
            return

        features = prepare_input(
            km,
            year,
            engine_size,
            brand,
            model,
            feature_names
        )

        if len(features) != len(parameters):
            print(
                f"Error: model expects {len(parameters)} parameters, "
                f"but input generated {len(features)} features."
            )
            return

        # T0, T1, T2 and T3 are stored using the scale transformation.
        # Dummy coefficients were incorrectly scaled during saving,
        # so compensate for that here until save_model() is fixed.
        parameters[4:] *= SCALE

        result = features @ parameters

        print(f"Estimated price: ${result:.2f}")

    except ValueError as error:
        print(f"Invalid input: {error}")
    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
