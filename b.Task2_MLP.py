# Imports
from EDF2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Global configuration
CLASS1_SIZE = 100
CLASS2_SIZE = 100
CLASS_SIZE_PER_GAUSSIAN = 50
LEARNING_RATE = 0.02
EPOCHS = 150
TEST_SIZE = 0.25

# XOR data generation parameters
# Class 0 (diagonal corners)
MEAN_CLASS0_1 = np.array([0, 0])
COV_CLASS0_1 = np.array([[0.3, 0], [0, 0.3]])
MEAN_CLASS0_2 = np.array([2, 2])
COV_CLASS0_2 = np.array([[0.3, 0], [0, 0.3]])

# Class 1 (opposite diagonal corners)
MEAN_CLASS1_1 = np.array([0, 2])
COV_CLASS1_1 = np.array([[0.5, 0], [0, 0.5]])
MEAN_CLASS1_2 = np.array([2, 0])
COV_CLASS1_2 = np.array([[0.5, 0], [0, 0.5]])

# Data generation
X_class0_g1 = multivariate_normal.rvs(MEAN_CLASS0_1, COV_CLASS0_1, CLASS_SIZE_PER_GAUSSIAN)
X_class0_g2 = multivariate_normal.rvs(MEAN_CLASS0_2, COV_CLASS0_2, CLASS_SIZE_PER_GAUSSIAN)
X1 = np.vstack((X_class0_g1, X_class0_g2))

X_class1_g1 = multivariate_normal.rvs(MEAN_CLASS1_1, COV_CLASS1_1, CLASS_SIZE_PER_GAUSSIAN)
X_class1_g2 = multivariate_normal.rvs(MEAN_CLASS1_2, COV_CLASS1_2, CLASS_SIZE_PER_GAUSSIAN)
X2 = np.vstack((X_class1_g1, X_class1_g2))

# Features and labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Train/test split
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model architecture
n_input = 2
n_hidden = 20
n_output = 1

# Parameter initialization (He for hidden layers, zeros for biases)
A1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
b1 = np.zeros((n_hidden, 1))
A2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros((n_hidden, 1))
A3 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
b3 = np.zeros((n_output, 1))

# Graph nodes
x_node = Input()
y_node = Input()
A1_node = Parameter(A1)
b1_node = Parameter(b1)
A2_node = Parameter(A2)
b2_node = Parameter(b2)
A3_node = Parameter(A3)
b3_node = Parameter(b3)

# Computation graph
linear1 = Linear(x_node, A1_node, b1_node)
sigmoid1 = Sigmoid(linear1)
linear2 = Linear(sigmoid1, A2_node, b2_node)
sigmoid2 = Sigmoid(linear2)
linear3 = Linear(sigmoid2, A3_node, b3_node)
output = Sigmoid(linear3)
loss = BCE(y_node, output)

# Static graph and trainable parameters
graph = [
    x_node, A1_node, b1_node, linear1, sigmoid1,
    A2_node, b2_node, linear2, sigmoid2,
    A3_node, b3_node, linear3, output, y_node, loss
]
trainable = [A1_node, b1_node, A2_node, b2_node, A3_node, b3_node]

# Training utilities
def forward_pass(graph_nodes):
    """Execute forward pass"""
    for n in graph_nodes:
        n.forward()

def backward_pass(graph_nodes):
    """Execute backward pass"""
    for n in graph_nodes[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    """SGD update to parameters."""
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

def train_with_batchsize(batch_size, epochs, lr):
    n_samples = X_train.shape[0]
    losses = []
    test_losses = []

    for epoch in range(epochs):
        # Shuffle each epoch
        perm = np.arange(n_samples)
        np.random.shuffle(perm)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]

        n_batches = int(np.ceil(n_samples / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)

            # Shapes: x_node=(2,B), y_node=(1,B)
            X_batch = X_shuffled[start:end].T
            y_batch = y_shuffled[start:end].reshape(1, -1)

            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, lr)

            # Accumulate batch loss
            epoch_loss += loss.value

        # Average loss per sample for the epoch
        losses.append(epoch_loss / n_samples)

        # Compute test loss
        x_node.value = X_test.T
        y_node.value = y_test.reshape(1, -1)
        forward_pass(graph)
        test_loss = loss.value
        test_losses.append(test_loss / len(y_test))

        if epoch % 10 == 0:
            print(f"Batch Size {batch_size} | Epoch {epoch + 1}/{epochs} | Loss: {losses[-1]:.4f}")

    return losses, test_losses

# Training across batch sizes
batch_sizes = [1, 2, 4, 8]
all_losses = {}
all_test_losses = {}

for bs in batch_sizes:
    print(f"\nTraining with batch size = {bs}")

    # Reinitialize parameters for each run to ensure fair comparison
    A1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
    b1 = np.zeros((n_hidden, 1))
    A2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((n_hidden, 1))
    A3 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
    b3 = np.zeros((n_output, 1))

    A1_node.value, b1_node.value = A1, b1
    A2_node.value, b2_node.value = A2, b2
    A3_node.value, b3_node.value = A3, b3

    losses, test_losses = train_with_batchsize(bs, EPOCHS, LEARNING_RATE)
    all_losses[bs] = losses
    all_test_losses[bs] = test_losses


# Final model evaluation using the last trained parameters
correct = 0
for i in range(X_test.shape[0]):
    x_node.value = X_test[i].reshape(2, 1)
    forward_pass(graph)
    pred = 1 if output.value >= 0.5 else 0
    correct += (pred == y_test[i])

accuracy = correct / X_test.shape[0]
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")


# Plot training loss comparison (AI generated)
plt.figure(figsize=(8, 6))
for bs, losses in all_losses.items():
    plt.plot(losses, label=f"Batch Size {bs}")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("MLP Training Loss (XOR Problem)")
plt.legend()
plt.grid(True)
plt.show()

# Plot test loss comparison (AI generated)
plt.figure(figsize=(8, 6))
for bs, test_losses in all_test_losses.items():
    plt.plot(test_losses, label=f"Batch Size {bs}")
plt.xlabel("Epoch")
plt.ylabel("Average Test Loss")
plt.title("MLP Test Loss (XOR Problem)")
plt.legend()
plt.grid(True)
plt.show()


# Decision boundary plot for the final model (AI generated)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 150),
    np.linspace(y_min, y_max, 150)
)

Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([[i], [j]])
    forward_pass(graph)
    Z.append(output.value[0, 0])

Z = np.array(Z).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap="RdYlBu", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="RdYlBu")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"MLP Decision Boundary (Accuracy: {accuracy * 100:.2f}%)")
plt.show()
