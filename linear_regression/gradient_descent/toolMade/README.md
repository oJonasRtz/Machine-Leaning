# Training with a NumPy tool

This version solves the same linear regression problem using the `numpy.polyfit` tool. It was created as an alternative to the manually implemented algorithm, allowing a custom implementation to be compared with a solution already available in the library.

The script reads the `km` and `price` columns from the CSV and passes these values to:

```python
numpy.polyfit(x_values, y_values, 1)
```

The third argument, `1`, indicates that the data should be fitted with a first-degree polynomial, meaning a straight line. The function returns the `T1` and `T0` coefficients, which correspond to the slope and intercept of the function:

```text
F(x) = T0 + T1 * x
```

The result is displayed in the terminal. Unlike the implementation in [`handMade`](../handMade/README.md), this script does not execute gradient descent step by step and does not save a checkpoint.

## Execution

From the `gradient_descent` directory:

```bash
python toolMade/train.py data.csv
```