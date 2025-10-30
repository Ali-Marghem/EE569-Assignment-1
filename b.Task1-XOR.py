# Imports
from EDF2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Global Configuration

# Data sizes and model hyperparameters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1

LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

# Gaussian mixture layout (XOR pattern)
CLASS_SIZE_PER_GAUSSIAN = 50  # 50 samples per Gaussian, 100 per class

# Class 0 Gaussians (diagonal corners)
MEAN_CLASS0_1 = np.array([0, 0])
COV_CLASS0_1 = np.array([[0.1, 0], [0, 0.1]])

MEAN_CLASS0_2 = np.array([2, 2])
COV_CLASS0_2 = np.array([[0.1, 0], [0, 0.1]])

# Class 1 Gaussians (opposite diagonal corners)
MEAN_CLASS1_1 = np.array([0, 2])
COV_CLASS1_1 = np.array([[0.1, 0], [0, 0.1]])

MEAN_CLASS1_2 = np.array([2, 0])
COV_CLASS1_2 = np.array([[0.1, 0], [0, 0.1]])


# Data Generation

# Class 0 samples (two Gaussians)
X_class0_g1 = multivariate_normal.rvs(MEAN_CLASS0_1, COV_CLASS0_1, CLASS_SIZE_PER_GAUSSIAN)
X_class0_g2 = multivariate_normal.rvs(MEAN_CLASS0_2, COV_CLASS0_2, CLASS_SIZE_PER_GAUSSIAN)
X1 = np.vstack((X_class0_g1, X_class0_g2))

# Class 1 samples (two Gaussians)
X_class1_g1 = multivariate_normal.rvs(MEAN_CLASS1_1, COV_CLASS1_1, CLASS_SIZE_PER_GAUSSIAN)
X_class1_g2 = multivariate_normal.rvs(MEAN_CLASS1_2, COV_CLASS1_2, CLASS_SIZE_PER_GAUSSIAN)
X2 = np.vstack((X_class1_g1, X_class1_g2))

# Combine features and labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))


# Train/Test Split

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


# Model Definition (Graph)

# Initialize linear parameters
W0 = np.zeros((1, 1))
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

# Arrange weights for Linear node: y = A @ x + b
A = np.array([W1[0], W2[0]]).reshape(1, 2)
b = W0

# Graph nodes: inputs, parameters, and operations
x_node = Input()
y_node = Input()
A_node = Parameter(A)
b_node = Parameter(b)

linear_node = Linear(x_node, A_node, b_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

# Static graph definition
graph = [x_node, A_node, b_node, linear_node, sigmoid, loss]
trainable = [A_node, b_node]


# Training Utilities
def forward_pass(graph_nodes):
    """Run forward pass over all nodes"""
    for n in graph_nodes:
        n.forward()

def backward_pass(graph_nodes):
    """Run backward pass over all nodes"""
    for n in graph_nodes[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

def train_with_batchsize(batch_size, epochs=EPOCHS, lr=LEARNING_RATE):
    losses = []
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_samples / batch_size))

        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, n_samples)

            # Node shapes: x_node expects (features, batch), y_node expects (1, batch)
            X_batch = X_train[start:end].T                # (2, B)
            y_batch = y_train[start:end].reshape(1, -1)   # (1, B)

            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, lr)

            # Accumulate batch loss (scalar)
            epoch_loss += loss.value

        # Average loss per sample for this epoch (consistent scalar normalization)
        losses.append(epoch_loss / n_samples)

        if epoch % 10 == 0:
            print(f"Batch Size {batch_size} | Epoch {epoch + 1}/{epochs} | Loss: {losses[-1]:.4f}")

    return losses


# Training Across Batch Sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
all_losses = {}

for bs in batch_sizes:
    print(f"\nTraining with batch size = {bs}")

    # Reinitialize parameters for each run to ensure fair comparison
    A_node.value = np.array([np.random.randn(), np.random.randn()]).reshape(1, 2) * 0.1
    b_node.value = np.zeros((1, 1))

    losses = train_with_batchsize(bs, EPOCHS, LEARNING_RATE)
    all_losses[bs] = losses


# Evaluation Helper
def evaluate_accuracy(X_eval, y_eval):
    """Compute classification accuracy using threshold 0.5 on sigmoid output."""
    correct = 0
    for i in range(X_eval.shape[0]):
        x_node.value = X_eval[i].reshape(2, 1)
        forward_pass(graph)
        pred = 1 if sigmoid.value[0, 0] >= 0.5 else 0
        if pred == y_eval[i]:
            correct += 1
    return correct / X_eval.shape[0]

# ==============================
# Results: Accuracy and Loss Curves (AI generated)
# ==============================
accuracy = evaluate_accuracy(X_test, y_test)
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 6))
for bs, losses in all_losses.items():
    plt.plot(losses, label=f"Batch Size {bs}")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Logistic Regression Training Loss (XOR Problem)")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# Decision Boundary Plot (AI generated)
# ==============================
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()

xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))

# Score grid points through the graph
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
