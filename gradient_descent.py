from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(x):
    return x**2.0

# Derivative (gradient)
def derivative(x):
    return 2.0 * x

print("Objective at 10:", objective(10))

bounds = np.array([[-10.0, 10.0]])

# Generate random samples for visualization
inputs = []
outputs = []

for _ in range(1000):
    x_sample = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    inputs.append(x_sample)
    outputs.append(objective(x_sample))

# Flatten inputs/outputs for plotting
inputs = np.array(inputs).flatten()
outputs = np.array(outputs).flatten()

# Gradient descent initialization
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
x = x[0]  # convert to scalar for safe calculations

solutions = []
evaluations = []

# Gradient descent loop
learning_rate = 0.1
for i in range(20):
    solutions.append(x)
    x_evaluation = objective(x)
    evaluations.append(x_evaluation)
    
    gradient = derivative(x)
    x = x - learning_rate * gradient
    
    # Clean print using scalars
    print(f'>{i} f({x:.5f}) = {x_evaluation:.5f}')

# Plot
plt.scatter(inputs, outputs, alpha=0.3, label="Random Samples")
plt.scatter(solutions, evaluations, color='red', label="Gradient Descent Steps")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent Visualization")
plt.legend()
plt.show()
