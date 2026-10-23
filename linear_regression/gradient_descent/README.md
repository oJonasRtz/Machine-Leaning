# Gradient descent

This project demonstrates how to use *gradient descent* to fit a linear regression model that estimates a car's price from its mileage.

The model is defined by the following function:

```text
F(x) = T0 + T1 * x
```

In this case, `x` is the car's mileage, `T0` is the intercept, and `T1` represents the estimated change in price as mileage changes. The training data is available in [`data.csv`](./data.csv).

## Prediction

The [`service.py`](./service.py) program is responsible for making predictions. It receives the mileage from the command line, loads `T0` and `T1` from [`checkpoint.json`](./checkpoint.json), and applies the model function.

To run it:

```bash
python service.py 150000
```

The `checkpoint.json` file stores the parameters learned during training, along with information about the process, such as the number of epochs and the MSE and RMSE metrics. The service does not train the model; it only uses the saved parameters to make predictions.

## Index

- [Manual training implementation](./handMade/README.md)
- [Implementation using a NumPy tool](./toolMade/README.md)
- [Extended multi-parameter implementation](./extend/README.md)