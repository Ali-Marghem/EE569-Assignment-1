import time
from cTask_2_EDF4 import *
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Timer start
total_start = time.perf_counter()

# Experiment configuration
TRAIN_N = 60000
TEST_N  = 10000
PREVIEW_N = 9

# Hyperparameters
n_output = 10
LEARNING_RATE = 0.0005
EPOCHS = 11
BATCH_SIZE = 128
LAMBDA_REG = 0.001
initialization = 'He'
np.random.seed(25)

# Load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()
if TRAIN_N is not None:
    train_X, train_y = train_X[:TRAIN_N], train_y[:TRAIN_N]
if TEST_N is not None:
    test_X, test_y = test_X[:TEST_N], test_y[:TEST_N]

print('X_train:', train_X.shape)
print('Y_train:', train_y.shape)
print('X_test: ', test_X.shape)
print('Y_test: ', test_y.shape)

# Preview
plt.figure(figsize=(4,4))
nprev = min(PREVIEW_N, len(train_X))
for i in range(nprev):
    r = int(np.ceil(np.sqrt(nprev)))
    c = int(np.ceil(nprev / r))
    plt.subplot(r, c, i+1); plt.imshow(train_X[i], cmap='gray'); plt.axis('off')
plt.suptitle('Sample training images'); plt.tight_layout(); plt.show()

# Preprocessing
X_train_img = (train_X.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
X_test_img  = (test_X.astype(np.float32)  / 255.0).reshape(-1, 1, 28, 28)

def one_hot_encode(y, num_classes=10):
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

y_train = one_hot_encode(train_y, 10)  # (N,10)
y_test  = one_hot_encode(test_y, 10)   # (N,10)
y_train_labels = train_y
y_test_labels  = test_y

# Training utilities
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]


# Build CNN
x_node = Input()
y_node = Input()

c1 = Conv(x_node, in_channels=1,  out_channels=16, kernel_size=3, initialization=initialization)
a1 = ReLU(c1)
p1 = MaxPooling(a1, pool_size=2, stride=2)

c2 = Conv(p1,   in_channels=16, out_channels=32, kernel_size=3, initialization=initialization)
a2 = ReLU(c2)
p2 = MaxPooling(a2, pool_size=2, stride=2)

c3 = Conv(p2,   in_channels=32, out_channels=64, kernel_size=3, initialization=initialization)
a3 = ReLU(c3)
p3 = MaxPooling(a3, pool_size=2, stride=2)

c4 = Conv(p3,   in_channels=64, out_channels=128, kernel_size=1, initialization=initialization)
a4 = ReLU(c4)

flat = Flatten(a4)  # -> (128,B)
fc   = Linear(flat, n_output=n_output, n_input=128, initialization=initialization, lambda_reg=LAMBDA_REG)
probs = Softmax(fc)
loss  = CrossEntropy(y_node, probs)

graph = topological_sort(loss)
trainables = collect_trainable_parameters(graph)


# Train/Evaluate

train_losses = []
test_losses  = []

N = X_train_img.shape[0]
for epoch in range(EPOCHS):
    # shuffle indices each epoch
    idx = np.arange(N)
    np.random.shuffle(idx)
    Xs, Ys = X_train_img[idx], y_train[idx]

    n_batches = int(np.ceil(N / BATCH_SIZE))
    epoch_loss = 0.0

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, N)
        Xb = Xs[s:e]
        Yb = Ys[s:e]

        x_node.value = Xb                 # (B,1,28,28)
        y_node.value = Yb.T               # (10,B)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, LEARNING_RATE)

        epoch_loss += loss.value   # summed CE over batch

    # average train loss per sample
    train_loss = epoch_loss / N
    train_losses.append(train_loss)

    # test loss
    x_node.value = X_test_img
    y_node.value = y_test.T
    forward_pass(graph)
    test_loss = loss.value / X_test_img.shape[0]
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# Final accuracy on test set
x_node.value = X_test_img
y_node.value = y_test.T
forward_pass(graph)
pred = np.argmax(probs.value, axis=0)
final_accuracy = np.mean(pred == y_test_labels)
print(f"Final CNN Accuracy: {final_accuracy*100:.2f}%")

# Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(train_losses, lw=2); plt.title('Train Loss'); plt.grid(True, alpha=0.3); plt.xlabel('Epoch')
plt.subplot(1,2,2); plt.plot(test_losses, lw=2);  plt.title('Test Loss');  plt.grid(True, alpha=0.3); plt.xlabel('Epoch')
plt.tight_layout(); plt.show()

print(f"Total time: {time.perf_counter() - total_start:.2f} s")
