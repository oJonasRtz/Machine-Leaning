# Manual training

This version explicitly implements the *gradient descent* training algorithm without using a ready-made regression fitting function. It uses `pandas` to read the data and `numpy` to represent values and perform matrix operations.

## How the algorithm works

1. The script reads the `km` and `price` columns from the CSV file.
2. The mileage is normalized by dividing each value by `100000`. This scale helps keep the values used in gradient calculations within a more suitable range.
3. For each observation, a row with `[1, normalized_km]` is created. The first value corresponds to `T0`, and the second corresponds to `T1`.
4. The model calculates predictions through matrix multiplication:

   ```text
   prediction = [1, normalized_km] · [T0, T1]
   ```

5. Each prediction error is compared with the actual price. The cost function is the mean squared error (MSE).
6. The gradient is calculated from the input matrix and the errors. In each update, the parameters move in the opposite direction of the gradient:

   ```text
   parameters = parameters - learning_rate * gradient
   ```

7. The process is repeated until the improvement in MSE is smaller than the defined tolerance. At the end, the parameters and training information are saved in a checkpoint.

Training is performed in batches of epochs. By default, each round runs `1000` epochs with a learning rate of `0.01`, then checks whether the model has converged.

## Execution

From the `gradient_descent` directory, training can be run with:

```bash
python handMade/train.py data.csv
```

The script saves the model to `checkpoint.json` in the current directory. With the command above, the file is created in the `gradient_descent` directory and can be used by [`service.py`](../service.py) to make predictions.

## Analysis notebook

- [Linear regression analysis](../Linear_regression_demo.ipynb)