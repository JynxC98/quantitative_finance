# Linear Regression: Theory

Linear Regression is one of the simplest and most fundamental algorithms in machine learning. It assumes a linear relationship between the input features and the target variable. The objective is to find the best-fit line (or hyperplane in higher dimensions) that minimises the error between the predicted and actual target values.

## 1. The Main Linear Regression Function

The linear regression model predicts the target variable `y` as a linear combination of the input features `X` and the model parameters (weights `W` and bias `b`):

### Hypothesis

The hypothesis function for linear regression is:

\[
h_{\theta}(X) = X \cdot W + b
\]

Where:
- \( X \) is the matrix of input features with dimensions \( m \times n \), where \( m \) is the number of training examples and \( n \) is the number of features.
- \( W \) is the weight vector of dimensions \( n \times 1 \).
- \( b \) is the bias term (a scalar).
- \( h_{\theta}(X) \) represents the predicted values.

### Cost Function

The cost function used in linear regression is the Mean Squared Error (MSE), which quantifies the error between the predicted values \( \hat{y} \) and the actual target values \( y \):

\[
J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(X^{(i)}) - y^{(i)})^2
\]

Where:
- \( m \) is the number of training examples.
- \( y^{(i)} \) and \( X^{(i)} \) are the actual target value and input features for the \( i \)-th training example.
- The factor \( \frac{1}{2} \) is added to simplify the derivative calculations.

---

## 2. The Propagation Function

In linear regression, we employ **forward propagation** to compute the predictions and **backward propagation** (also called gradient descent) to update the parameters.

### Forward Propagation

The forward propagation is the process of computing the predicted values using the hypothesis function:

\[
\hat{y} = X \cdot W + b
\]

Where:
- \( \hat{y} \) represents the predicted values (also called outputs).

### Loss Calculation

The loss for a single training example is the squared error between the predicted and actual target values:

\[
\text{Loss} = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(X^{(i)}) - y^{(i)})^2
\]

---

## 3. Derivative and Gradient Descent

To minimise the cost function \( J(W, b) \), we use **Gradient Descent**, an optimisation technique that iteratively adjusts the weights \( W \) and bias \( b \) in the direction of the steepest descent of the cost function.

### Gradient of the Cost Function

The gradients of the cost function with respect to the weight vector \( W \) and bias term \( b \) are computed as follows:

1. **Derivative with respect to \( W \)**:

\[
\frac{\partial J(W, b)}{\partial W} = \frac{1}{m} X^T (h_{\theta}(X) - y)
\]

2. **Derivative with respect to \( b \)**:

\[
\frac{\partial J(W, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(X^{(i)}) - y^{(i)})
\]

Where:
- \( X^T \) is the transpose of the input matrix \( X \).
- The term \( h_{\theta}(X) - y \) is the difference between the predicted and actual target values.

### Parameter Update Rule

The weight vector \( W \) and the bias term \( b \) are updated simultaneously using the following gradient descent update rules:

\[
W := W - \alpha \frac{1}{m} X^T (h_{\theta}(X) - y)
\]

\[
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(X^{(i)}) - y^{(i)})
\]

Where:
- \( \alpha \) is the learning rate, a hyperparameter that controls the step size of each update.

---

## Summary of Linear Regression Workflow

1. **Initialize** the parameters \( W \) and \( b \) to small random values.
2. **Forward Propagation**: Compute the predictions using the hypothesis \( h_{\theta}(X) = X \cdot W + b \).
3. **Compute the Cost**: Calculate the mean squared error between predictions and actual values.
4. **Backward Propagation**: Compute the gradients with respect to \( W \) and \( b \).
5. **Gradient Descent**: Update the parameters \( W \) and \( b \) using the gradients.
6. **Iterate** until convergence (i.e., when the cost function is minimized).

By following this process, linear regression learns the best-fit line for a given dataset.

---
