
# Extended multi-parameter gradient descent

This project extends the previous gradient descent experiments to use multiple input features. While the other training implementations are limited to a linear model with two parameters (`T0` and `T1`), this version trains a model with 12 parameters: one intercept and 11 feature coefficients.

## Model

The model estimates a car's price using the following input features:

- `km`: the car's mileage
- `year`: the manufacturing year
- `engine_size`: the engine size
- `brand`: the car brand
- `model`: the car model

The numeric features are scaled before training. The categorical `brand` and `model` values are converted into numeric dummy features using one-hot encoding. The first category in each group is used as the reference category, so the resulting feature matrix contains:

```text
[1, km, year, engine_size, brand features, model features]
```

The `1` is the intercept term (`T0`). Together with the three numeric features, three brand features, and seven model features, this produces 12 model parameters:

```text
F(x) = T0 + T1*x1 + T2*x2 + ... + T11*x11
```

The complete list of feature names is stored in the training section of [`checkpoint.json`](./checkpoint.json), so the prediction service can recreate the same feature order used during training.

## Training

The [`train.py`](./train.py) script reads [`dataset.csv`](./dataset.csv), prepares the numeric and categorical features, and initializes one parameter for each feature. It then applies gradient descent to minimize the mean squared error:

```text
parameters = parameters - learning_rate * gradient
```

Training runs in batches of epochs and stops when the improvement in loss is below the configured tolerance. The learned parameters, training metrics, and feature names are saved to [`checkpoint.json`](./checkpoint.json).

From this directory, run:

```bash
python train.py dataset.csv
```

## Prediction

The [`service.py`](./service.py) script loads the saved parameters and feature names from `checkpoint.json`. It accepts the same types of information used during training, validates the categorical values, rebuilds the feature vector in the correct order, and calculates the estimated price.

Run it with:

```bash
python service.py 150000 2012 1.4 Fiat Palio
```

The service does not train the model. It uses the parameters already stored in the checkpoint to make a prediction.

## Comparison with scikit-learn

The [`sklearn_train.py`](./sklearn_train.py) script trains the same type of multi-feature linear regression using `LinearRegression` from scikit-learn, an established machine learning library.

It is included as a reference implementation for comparing the gradient descent algorithm developed in [`train.py`](./train.py) with a library-provided solution. Both scripts use the same dataset and features, including mileage, year, engine size, brand, and model, and report the resulting coefficients, MSE, and RMSE.

The scikit-learn script is intended for study and comparison. It does not replace the custom implementation, does not perform the manual gradient descent steps, and does not save a checkpoint for the prediction service.

From this directory, run:

```bash
python sklearn_train.py dataset.csv
```
