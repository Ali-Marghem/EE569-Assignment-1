# Imports
from EDF2_V2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Global configuration
CLASS1_SIZE = 100
CLASS2_SIZE = 100
LEARNING_RATE = 0.2
EPOCHS = 150
TEST_SIZE = 0.25
CLASS_SIZE_PER_GAUSSIAN = 50

# XOR data generation parameters
# Class 0 (diagonal corners)
MEAN_CLASS0_1 = np.array([0, 0])
COV_CLASS0_1 = np.array([[0.3, 0], [0, 0.3]])
MEAN_CLASS0_2 = np.array([2, 2])
COV_CLASS0_2 = np.array([[0.3, 0], [0, 0.3]])

# Class 1 (opposite diagonal corners)
MEAN_CLASS1_1 = np.array([0, 2])
COV_CLASS1_1 = np.array([[0.3, 0], [0, 0.3]])
MEAN_CLASS1_2 = np.array([2, 0])
COV_CLASS1_2 = np.array([[0.3, 0], [0, 0.3]])

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

# Model dimensions and initialization scheme
n_input = 2
n_hidden = 20
n_output = 1
initialization = 'He'

# Placeholders
x_node = Input()
y_node = Input()

# Computation graph with auto-parameter creation in Linear nodes
linear1 = Linear(x_node, n_hidden, n_input, initialization)
sigmoid1 = Sigmoid(linear1)
linear2 = Linear(sigmoid1, n_hidden, n_hidden, initialization)
sigmoid2 = Sigmoid(linear2)
linear3 = Linear(sigmoid2, n_output, n_hidden, initialization)
output = Sigmoid(linear3)
loss = BCE(y_node, output)

# Graph and trainables (topologically sorted)
graph = topological_sort(loss)
trainable = collect_trainable_parameters(graph)

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
    """SGD updates to trainable parameters."""
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

def train_with_batchsize(batch_size, epochs, lr):
    """
    Train the MLP using a given batch size.
    Returns lists of average train loss and average test loss per epoch.
    """
    n_samples = X_train.shape[0]
    losses = []
    test_losses = []

    for epoch in range(epochs):
        # Shuffle data each epoch
        perm = np.arange(n_samples)
        np.random.shuffle(perm)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]
        n_batches = int(np.ceil(n_samples / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)

            # Shapes for EDF2_V2: inputs are (features, batch), labels are (1, batch)
            X_batch = X_shuffled[start:end].T
            y_batch = y_shuffled[start:end].reshape(1, -1)

            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, lr)

            # Accumulate batch loss
            epoch_loss += loss.value

        # Average train loss per sample for the epoch
        losses.append(epoch_loss / n_samples)

        # Test loss
        x_node.value = X_test.T
        y_node.value = y_test.reshape(1, -1)
        forward_pass(graph)
        test_loss = loss.value
        test_losses.append(test_loss / len(y_test))

        if epoch % 10 == 0:
            print(f"Batch Size {batch_size} | Epoch {epoch + 1}/{epochs} | Loss: {losses[-1]:.4f}")

    return losses, test_losses

# Training across batch sizes
batch_sizes = [1, 2, 4, 8, 16]
all_losses = {}
all_test_losses = {}

for bs in batch_sizes:
    print(f"\nTraining with batch size = {bs}")

    # Reinitialize parameters for each batch size

    for param in trainable:

        if param.value.shape[0] > 1:  # Weight matrix

            if initialization == 'He':

                param.value = np.random.randn(*param.value.shape) * np.sqrt(2.0 / param.value.shape[1])

            elif initialization == 'zeros':

                param.value = np.zeros(*param.value.shape)

            else:

                param.value = np.random.randn(*param.value.shape) * 0.1

        else:  # Bias vector

            param.value = np.zeros(param.value.shape)

    losses, test_losses = train_with_batchsize(bs, EPOCHS, LEARNING_RATE)
    all_losses[bs] = losses
    all_test_losses[bs] = test_losses

# Final evaluation using the last trained parameters
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


# Decision boundary for the final model (AI generated)
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
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"MLP Decision Boundary (Accuracy: {accuracy * 100:.2f}%)")
plt.show()
