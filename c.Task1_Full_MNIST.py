# Imports
import time
from EDF3 import *
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Timer start
total_start = time.perf_counter()

# Data loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test: ' + str(test_X.shape))
print('Y_test: ' + str(test_y.shape))

# Quick visual check of samples
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
plt.show()

# Data preprocessing
# Flatten 28x28 images to 784-dimensional vectors
X_train = train_X.reshape(train_X.shape[0], -1)
X_test = test_X.reshape(test_X.shape[0], -1)

# Normalize to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot labels
def one_hot_encode(y, num_classes=10):
    """One-hot encode integer labels to shape (N, C)."""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_train = one_hot_encode(train_y, 10)
y_test = one_hot_encode(test_y, 10)
y_train_labels = train_y
y_test_labels = test_y
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Model hyperparameters
n_input = 784           # Full MNIST input size
n_hidden = 128          # Hidden layer width
n_output = 10           # 10 classes
LEARNING_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 128
LAMBDA_REG = 0.001
initialization = 'He'
np.random.seed(25)

# training utilities
def forward_pass(graph):
    """Run forward pass"""
    for n in graph:
        n.forward()

def backward_pass(graph):
    """Run backward pass"""
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

# Train a single configuration (activation + L2)
def train_model(activation_func, activation_name, lambda_reg=0.0, epochs=EPOCHS):
    # Build graph
    x_node = Input()
    y_node = Input()

    # Network: Linear -> Activation -> Linear -> Softmax
    linear1 = Linear(x_node, n_hidden, n_input, initialization, lambda_reg)
    activation1 = activation_func(linear1)
    linear2 = Linear(activation1, n_output, n_hidden, initialization, lambda_reg)
    softmax_output = Softmax(linear2)
    loss = CrossEntropy(y_node, softmax_output)

    # Topological order and parameters
    graph = topological_sort(loss)
    trainable = collect_trainable_parameters(graph)

    train_losses = []
    test_losses = []
    n_samples = X_train.shape[0]

    print(f"Training with {activation_name} (λ={lambda_reg})...")
    for epoch in range(epochs):
        # Shuffle each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]


        n_batches = int(np.ceil(n_samples / BATCH_SIZE))
        epoch_loss = 0

        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_samples)


            X_batch = X_shuffled[start:end].T
            y_batch = y_shuffled[start:end].T

            x_node.value = X_batch
            y_node.value = y_batch

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, LEARNING_RATE)

            # Accumulate per-batch summed loss
            epoch_loss += loss.value

        # Average training loss per sample
        train_loss = epoch_loss / n_samples
        train_losses.append(train_loss)

        # Test loss
        x_node.value = X_test.T
        y_node.value = y_test.T
        forward_pass(graph)
        test_loss = loss.value / X_test.shape[0]
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f" Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Final accuracy
    x_node.value = X_test.T
    y_node.value = y_test.T
    forward_pass(graph)
    predictions = np.argmax(softmax_output.value, axis=0)
    final_accuracy = np.sum(predictions == y_test_labels) / X_test.shape[0]
    print(f" Final {activation_name} Accuracy: {final_accuracy * 100:.2f}%\n")

    return train_losses, test_losses, final_accuracy

# Activation comparison without L2
print("\n" + "=" * 80)
print("ACTIVATION FUNCTION COMPARISON (No L2 Regularization)")
print("=" * 80)

activation_functions = [
    (Sigmoid, 'Sigmoid'),
    (ReLU, 'ReLU'),
    (Tanh, 'Tanh')
]
activation_results = {}

for activation_func, activation_name in activation_functions:
    train_losses, test_losses, accuracy = train_model(activation_func, activation_name, lambda_reg=0.0)
    activation_results[activation_name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }

# L2 regularization sweep for each activation
print("\n" + "=" * 80)
print("L2 REGULARIZATION EFFECT FOR ALL ACTIVATION FUNCTIONS")
print("=" * 80)

lambda_values = [0.0, 0.001, 0.01, 0.1]
l2_results = {}

for activation_func, activation_name in activation_functions:
    print(f"\n--- Testing L2 Regularization with {activation_name} ---")
    l2_results[activation_name] = {}
    for lambda_reg in lambda_values:
        train_losses, test_losses, accuracy = train_model(
            activation_func,
            f'{activation_name} (λ={lambda_reg})',
            lambda_reg=lambda_reg
        )
        l2_results[activation_name][f'λ={lambda_reg}'] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracy': accuracy
        }

# Plotting (AI generated)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# Top row: Activation comparison (no L2)
ax1 = fig.add_subplot(gs[0, 0])
colors_act = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (name, results) in enumerate(activation_results.items()):
    ax1.plot(results['train_losses'], label=name, linewidth=2, color=colors_act[i])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss: Activation Functions (No L2)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
for i, (name, results) in enumerate(activation_results.items()):
    ax2.plot(results['test_losses'], label=name, linewidth=2, color=colors_act[i])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Loss')
ax2.set_title('Test Loss: Activation Functions (No L2)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
names = list(activation_results.keys())
accuracies = [activation_results[name]['accuracy'] * 100 for name in names]
bars = ax3.bar(names, accuracies, color=colors_act)
ax3.set_ylabel('Test Accuracy (%)')
ax3.set_title('Final Accuracy: Activation Functions (No L2)')
ax3.set_ylim([0, 105])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
ax3.grid(True, axis='y', alpha=0.3)

ax4 = fig.add_subplot(gs[0, 3])
best_act = max(activation_results.items(), key=lambda x: x[1]['accuracy'])
worst_act = min(activation_results.items(), key=lambda x: x[1]['accuracy'])
ax4.plot(best_act[1]['test_losses'], label=f'Best: {best_act[0]}', linewidth=3, color='green')
ax4.plot(worst_act[1]['test_losses'], label=f'Worst: {worst_act[0]}', linewidth=3, color='red')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Test Loss')
ax4.set_title('Best vs Worst Activation Function')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Rows 2-4: L2 effects per activation
colors_l2 = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
comparison_colors = [
    ('#FF6B6B', '#4ECDC4'),
    ('#45B7D1', '#FFD700'),
    ('#99FF99', '#FF69B4')
]

for row, (activation_func, activation_name) in enumerate(activation_functions):
    ax_train = fig.add_subplot(gs[row + 1, 0])
    for i, (lambda_name, results) in enumerate(l2_results[activation_name].items()):
        ax_train.plot(results['train_losses'], label=lambda_name, linewidth=2, color=colors_l2[i])
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Training Loss')
    ax_train.set_title(f'Training Loss: {activation_name} with L2')
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_test = fig.add_subplot(gs[row + 1, 1])
    for i, (lambda_name, results) in enumerate(l2_results[activation_name].items()):
        ax_test.plot(results['test_losses'], label=lambda_name, linewidth=2, color=colors_l2[i])
    ax_test.set_xlabel('Epoch')
    ax_test.set_ylabel('Test Loss')
    ax_test.set_title(f'Test Loss: {activation_name} with L2')
    ax_test.legend()
    ax_test.grid(True, alpha=0.3)

    ax_acc = fig.add_subplot(gs[row + 1, 2])
    l2_names = list(l2_results[activation_name].keys())
    l2_accuracies = [l2_results[activation_name][name]['accuracy'] * 100 for name in l2_names]
    bars = ax_acc.bar(l2_names, l2_accuracies, color=colors_l2)
    ax_acc.set_ylabel('Test Accuracy (%)')
    ax_acc.set_title(f'Final Accuracy: {activation_name} with L2')
    ax_acc.set_ylim([0, 105])
    for bar, acc in zip(bars, l2_accuracies):
        height = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width() / 2., height + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    ax_acc.grid(True, axis='y', alpha=0.3)

    ax_comp = fig.add_subplot(gs[row + 1, 3])
    no_l2_result = activation_results[activation_name]
    best_l2_result = max(l2_results[activation_name].items(), key=lambda x: x[1]['accuracy'])
    ax_comp.plot(no_l2_result['test_losses'], label=f'{activation_name} (No L2)', linewidth=2, color=comparison_colors[row][0])
    ax_comp.plot(best_l2_result[1]['test_losses'], label=f'{activation_name} ({best_l2_result[0]})', linewidth=2, color=comparison_colors[row][1])
    ax_comp.set_xlabel('Epoch')
    ax_comp.set_ylabel('Test Loss')
    ax_comp.set_title(f'{activation_name}: Best L2 vs No L2')
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)

plt.suptitle('MNIST Digit Classification: Complete Analysis - Activation Functions & L2 Regularization', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Timer end
print(f"Total time: {time.perf_counter() - total_start:.2f} s")
