# Logistic Regression: Theory

Logistic Regression is a classification algorithm used to predict the probability of a binary outcome (e.g., 0 or 1). Unlike linear regression, which predicts continuous values, logistic regression applies the logistic (sigmoid) function to model a target variable that can only take two discrete values.

## 1. The Main Logistic Regression Function

The logistic regression model uses the following hypothesis to estimate the probability that a given input belongs to class 1:

### Hypothesis

\[
h_{\theta}(X) = \sigma(X \cdot W + b)
\]

Where:
- \( X \) is the matrix of input features, with dimensions \( m \times n \), where \( m \) is the number of training examples and \( n \) is the number of features.
- \( W \) is the weight vector of dimensions \( n \times 1 \).
- \( b \) is the bias term (a scalar).
- \( h_{\theta}(X) \) represents the predicted probability that the input belongs to class 1.
- \( \sigma(z) \) is the **sigmoid function**, which maps any real-valued number to a value between 0 and 1.

The sigmoid function is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where \( z = X \cdot W + b \). The output of this function represents the probability that the input belongs to class 1. If \( \sigma(z) \geq 0.5 \), we classify the input as class 1; otherwise, it belongs to class 0.

### Cost Function

Logistic regression uses the **Log Loss** (also called binary cross-entropy) as its cost function, which measures how well the model's predictions match the true labels:

\[
J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(X^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(X^{(i)})) \right]
\]

Where:
- \( m \) is the number of training examples.
- \( y^{(i)} \) is the actual label for the \( i \)-th example.
- \( h_{\theta}(X^{(i)}) \) is the predicted probability for the \( i \)-th example.

The goal is to minimise this cost function to improve the accuracy of the predictions.

---

## 2. The Propagation Function

In logistic regression, we employ **forward propagation** to compute the predicted probabilities and **backward propagation** (gradient descent) to update the model parameters.

### Forward Propagation

The forward propagation step involves applying the hypothesis function \( h_{\theta}(X) = \sigma(X \cdot W + b) \) to compute the predicted probabilities for each training example.

### Loss Calculation

For each training example, the loss is given by:

\[
\text{Loss} = - \left[ y \log(h_{\theta}(X)) + (1 - y) \log(1 - h_{\theta}(X)) \right]
\]

This measures how close the predicted probability is to the actual class label.

---

## 3. Derivative and Gradient Descent

To minimise the cost function \( J(W, b) \), we use **Gradient Descent**, an iterative optimisation algorithm that updates the weights \( W \) and bias \( b \) in the direction of the steepest descent of the cost function.

### Gradient of the Cost Function

The gradients of the cost function with respect to the weight vector \( W \) and the bias term \( b \) are computed as follows:

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
- \( h_{\theta}(X) - y \) is the vector of prediction errors (difference between the predicted and actual values).

### Parameter Update Rule

The parameters are updated using the gradient descent algorithm as follows:

\[
W := W - \alpha \frac{1}{m} X^T (h_{\theta}(X) - y)
\]

\[
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(X^{(i)}) - y^{(i)})
\]

Where:
- \( \alpha \) is the learning rate, which controls the step size of each update.

---

## Summary of Logistic Regression Workflow

1. **Initialise** the parameters \( W \) and \( b \) to small random values.
2. **Forward Propagation**: Compute the predicted probabilities using the hypothesis function \( h_{\theta}(X) = \sigma(X \cdot W + b) \).
3. **Compute the Cost**: Calculate the binary cross-entropy loss between the predicted and actual values.
4. **Backward Propagation**: Compute the gradients with respect to \( W \) and \( b \).
5. **Gradient Descent**: Update the parameters \( W \) and \( b \) using the calculated gradients.
6. **Repeat** until the cost function converges.

This process allows the logistic regression model to learn the decision boundary that best separates the two classes.

---
