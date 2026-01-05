# logistic-regression-with-only-numpy
Logistic regression implemented from first principles using NumPy and vector calculus.

## Mathematical Background

Given data X ∈ ℝ^{n×d}, weights w ∈ ℝ^d, and bias b:

Sigmoid:
σ(z) = 1 / (1 + exp(-z))

Model:
ŷ = σ(Xw + b)

Binary Cross-Entropy Loss:
L = -(1/n) Σ [ y log(ŷ) + (1 - y) log(1 - ŷ) ]

Gradients:
∂L/∂w = (1/n) Xᵀ(ŷ - y)
∂L/∂b = (1/n) Σ (ŷ - y)
