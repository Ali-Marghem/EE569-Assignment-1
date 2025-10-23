from Task1_LCN import *
from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

# Reshape into matrix form for Linear node
A = np.array([W1[0], W2[0]]).reshape(1, 2)
b = W0

# Create nodes
x_node = Input()
y_node = Input()
A_node = Parameter(A)
b_node = Parameter(b)

# Build computation graph using Linear node
linear_node = Linear(x_node, A_node, b_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node, A_node, b_node, linear_node, sigmoid, loss]
trainable = [A_node, b_node]

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()


def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]


# Task 3: Training with Batching
def train_with_batchsize(batch_size, epochs=EPOCHS, lr=0.02):
    losses = []
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = int(np.ceil(n_samples / batch_size))

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_train[start:end].T  # (2, batch_size)
            y_batch = y_train[start:end].reshape(1, -1)  # (1, batch_size)

            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, lr)

            # Accumulate loss (loss.value is sum over batch)
            epoch_loss += loss.value

        # Calculate average loss per sample for this epoch
        losses.append(epoch_loss / n_samples)

        print(f"Batch Size {batch_size} | Epoch {epoch + 1}/{epochs} | Loss: {losses[-1]:.4f}")

    return losses

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
all_losses = {}

for bs in batch_sizes:
    # Reset model parameters before each run
    A_node.value = np.array([np.random.randn(), np.random.randn()]).reshape(1, 2) * 0.1
    b_node.value = np.zeros(1)
    print(f"\nTraining with batch size = {bs}")
    all_losses[bs] = train_with_batchsize(bs, epochs=EPOCHS, lr=LEARNING_RATE)


# Task4: Plot the Training Loss vs. Epoch for Different Batch Sizes

plt.figure(figsize=(8, 6))
for bs, losses in all_losses.items():
    plt.plot(losses, label=f"Batch Size {bs}")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Effect of Batch Size on Training Loss")
plt.legend()
plt.grid(True)
plt.show()


# Model Evaluation (using final trained model)
correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.value = X_test[i].reshape(2, 1)
    forward_pass(graph)

    if sigmoid.value[0, 0] >= 0.5:
        prediction = 1
    else:
        prediction = 0

    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")

# Decision Boundary Visualization
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []

for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([i, j]).reshape(2, 1)
    forward_pass(graph)
    Z.append(sigmoid.value[0, 0])

Z = np.array(Z).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary with Linear Node (Final Model)")
plt.show()
