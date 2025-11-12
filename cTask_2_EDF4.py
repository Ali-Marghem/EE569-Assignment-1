import numpy as np


# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]


# ReLU Activation Node
class ReLU(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        self.value = np.maximum(0, input_value)

    def backward(self):
        input_value = self.inputs[0].value
        partial = (input_value > 0).astype(float)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

# Tanh Activation Node
class Tanh(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        self.value = np.tanh(input_value)

    def backward(self):
        partial = 1 - self.value ** 2
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

# Softmax Activation Node
class Softmax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        exp_values = np.exp(input_value - np.max(input_value, axis=0, keepdims=True))
        self.value = exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        self.gradients[self.inputs[0]] = grad_output


class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value*np.log(y_pred.value)-(1-y_true.value)*np.log(1-y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1-y_pred.value))


# Cross-Entropy Loss
class CrossEntropy(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        # y_true: one-hot encoded (n_classes, batch_size)
        # y_pred: softmax output (n_classes, batch_size)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred.value, 1e-10, 1 - 1e-10)
        self.value = -np.sum(y_true.value * np.log(y_pred_clipped))


    def backward(self):
        y_true, y_pred = self.inputs
        # Gradient of cross-entropy with respect to softmax input
        self.gradients[y_pred] = (y_pred.value - y_true.value)
        self.gradients[y_true] = -np.log(np.clip(y_pred.value, 1e-10, 1 - 1e-10))


class Linear(Node):
    """Linear computation node: y = Ax + b"""
    def __init__(self, x, n_output, n_input, initialization='He', lambda_reg=0.0):
        """
        Initialize Linear node with automatic parameter creation
        lambda_reg: L2 regularization parameter
        """
        # Initialize weights
        if initialization.lower() == 'he':
            A = np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)
        elif initialization == 'zeros':
            A = np.zeros((n_output, n_input))
        else:
            A = np.random.randn(n_output, n_input) * 0.1

        # Initialize biases
        b = np.zeros((n_output, 1))

        # Create parameter nodes
        self.A = Parameter(A)
        self.b = Parameter(b)
        self.lambda_reg = lambda_reg

        # Initialize parent Node with all inputs
        Node.__init__(self, [x, self.A, self.b])

    def forward(self):
        """Compute forward pass: y = Ax + b"""
        x, A, b = self.inputs
        self.value = np.dot(A.value, x.value) + b.value

    def backward(self):
        """Compute gradients with L2 regularization"""
        x, A, b = self.inputs
        grad_output = self.outputs[0].gradients[self]

        # dy/dx = A^T @ grad_output
        self.gradients[x] = np.dot(A.value.T, grad_output)

        # dy/dA = grad_output @ x^T + L2 regularization term
        self.gradients[A] = np.dot(grad_output, x.value.T) + self.lambda_reg * A.value

        # dy/db = sum over batch dimension (keepdims for consistency)
        self.gradients[b] = np.sum(grad_output, axis=1, keepdims=True)


class Conv(Node):
    """
    Convolutional layer node that performs 2D .
    Input:  (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, out_height, out_width)
    where out_height = height - kernel_size + 1 and out_width = width - kernel_size + 1
    """

    def __init__(self, x, in_channels, out_channels, kernel_size=3, initialization='He'):
        # Initialize kernels/filters
        if initialization.lower() == 'he':
            kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
                2.0 / (in_channels * kernel_size * kernel_size))
        else:
            kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1

        # Initialize biases (one per output channel)
        biases = np.zeros((out_channels, 1))

        # Create parameter nodes
        self.kernels = Parameter(kernels)
        self.biases = Parameter(biases)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        Node.__init__(self, [x, self.kernels, self.biases])

    def forward(self):
        x, kernels, biases = self.inputs
        X = x.value  # (B, C_in, H, W)
        B, C_in, H, W = X.shape
        K = self.kernel_size
        H_out = H - K + 1
        W_out = W - K + 1

        out = np.zeros((B, self.out_channels, H_out, W_out), dtype=X.dtype)

        # Sliding-window convolution via tensordot over (C_in, K, K)
        for i in range(H_out):
            v_start = i
            v_end = i + K
            for j in range(W_out):
                h_start = j
                h_end = j + K
                window = X[:, :, v_start:v_end, h_start:h_end]  # (B, C_in, K, K)
                # (B, C_out) from sum over (C_in, K, K)
                out[:, :, i, j] = np.tensordot(window, kernels.value, axes=([1, 2, 3], [1, 2, 3]))

        # Add bias per output channel
        out += biases.value[:, 0][np.newaxis, :, np.newaxis, np.newaxis]
        self.value = out

    def backward(self):
        x, kernels, biases = self.inputs
        X = x.value  # (B, C_in, H, W)
        dY = self.outputs[0].gradients[self]  # (B, C_out, H_out, W_out)

        B, C_in, H, W = X.shape
        _, C_out, H_out, W_out = dY.shape
        K = self.kernel_size

        # Initialize grads
        self.gradients[x] = np.zeros_like(X)
        self.gradients[kernels] = np.zeros_like(kernels.value)
        self.gradients[biases] = np.zeros_like(biases.value)

        dX = self.gradients[x]              # (B, C_in, H, W)
        dK = self.gradients[kernels]        # (C_out, C_in, K, K)
        dB = self.gradients[biases]         # (C_out, 1)

        # Bias gradient: sum over batch and spatial positions
        dB[:, 0] = np.sum(dY, axis=(0, 2, 3))

        # Accumulate kernel and input gradients via tensordot over batch and channels
        for i in range(H_out):
            v_start = i
            v_end = i + K
            for j in range(W_out):
                h_start = j
                h_end = j + K

                window = X[:, :, v_start:v_end, h_start:h_end]  # (B, C_in, K, K)
                dY_ij = dY[:, :, i, j]                          # (B, C_out)

                # dK: sum over batch -> (C_out, C_in, K, K)
                dK[:] += np.tensordot(dY_ij, window, axes=([0], [0]))

                # dX window: sum over C_out -> (B, C_in, K, K)
                dX[:, :, v_start:v_end, h_start:h_end] += np.tensordot(dY_ij, kernels.value, axes=([1], [0]))


class MaxPooling(Node):
    """
    Max/Average pooling layer with cached masks for backward.
    Input:  (B, C, H, W)
    Output: (B, C, H_out, W_out), where
            H_out = (H - pool_size) // stride + 1
            W_out = (W - pool_size) // stride + 1
    """

    def __init__(self, x, pool_size=2, stride=None, mode='max'):
        Node.__init__(self, [x])
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mode = mode
        # Cache one-hot masks per (i, j) when mode == 'max'
        self._mask_cache = {}

    def forward(self):
        x = self.inputs[0]
        X = x.value  # (B, C, H, W)
        B, C, H, W = X.shape
        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        out = np.zeros((B, C, H_out, W_out), dtype=X.dtype)
        self._mask_cache.clear()

        for i in range(H_out):
            v_start = i * S
            v_end = v_start + P
            for j in range(W_out):
                h_start = j * S
                h_end = h_start + P

                window = X[:, :, v_start:v_end, h_start:h_end]  # (B, C, P, P)

                if self.mode == 'max':
                    out[:, :, i, j] = np.max(window, axis=(2, 3))
                    # Build one-hot mask of maxima per (B, C)
                    flat = window.reshape(B, C, P * P)
                    idx = np.argmax(flat, axis=2)  # (B, C)
                    mask = np.zeros_like(flat)
                    # Vectorized scatter
                    b_idx = np.arange(B)[:, None]
                    c_idx = np.arange(C)[None, :]
                    mask[b_idx, c_idx, idx] = 1
                    mask = mask.reshape(B, C, P, P)
                    self._mask_cache[(i, j)] = mask
                elif self.mode == 'average':
                    out[:, :, i, j] = np.mean(window, axis=(2, 3))
                else:
                    raise NotImplementedError("Invalid pooling mode: use 'max' or 'average'.")

        self.value = out

    def backward(self):
        x = self.inputs[0]
        X = x.value  # (B, C, H, W)
        dY = self.outputs[0].gradients[self]  # (B, C, H_out, W_out)

        B, C, H, W = X.shape
        _, _, H_out, W_out = dY.shape
        P = self.pool_size
        S = self.stride

        dX = np.zeros_like(X)

        for i in range(H_out):
            v_start = i * S
            v_end = v_start + P
            for j in range(W_out):
                h_start = j * S
                h_end = h_start + P

                if self.mode == 'max':
                    mask = self._mask_cache[(i, j)]  # (B, C, P, P)
                    # Broadcast upstream grad over window via mask
                    dX[:, :, v_start:v_end, h_start:h_end] += dY[:, :, i:i+1, j:j+1] * mask
                elif self.mode == 'average':
                    # Evenly distribute the gradient over the window
                    dX[:, :, v_start:v_end, h_start:h_end] += dY[:, :, i:i+1, j:j+1] / (P * P)
                else:
                    raise NotImplementedError("Invalid pooling mode: use 'max' or 'average'.")

        self.gradients[x] = dX

class Flatten(Node):
    """
    Flatten node to reshape from 4D (batch, channels, height, width)
    to 2D (features, batch).
    Input shape:  (B, C, H, W)
    Output shape: (C*H*W, B)
    """
    def __init__(self, x):
        Node.__init__(self, [x])
        self.input_shape = None

    def forward(self):
        x = self.inputs[0]
        self.input_shape = x.value.shape  # (B, C, H, W)
        B = self.input_shape[0]
        # Flatten per-sample, then transpose to (features, batch)
        self.value = x.value.reshape(B, -1).T

    def backward(self):
        x = self.inputs[0]
        # Upstream gradient w.r.t. this node
        grad_output = self.outputs[0].gradients[self]
        # Reshape back to original input shape
        self.gradients[x] = grad_output.T.reshape(self.input_shape)


def topological_sort(output_node):
    """
    Perform topological sort on computational graph using DFS

    Args:
        output_node: The final output node of the graph

    Returns:
        List of nodes in topological order
    """
    visited = set()
    graph = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)

        # Visit all parent nodes first (DFS)
        for parent in node.inputs:
            dfs(parent)

        # Add current node after all parents are added
        graph.append(node)

    dfs(output_node)
    return graph


def collect_trainable_parameters(graph):
    """
    Collect all Parameter nodes from the graph

    Args:
        graph: List of nodes in topological order

    Returns:
        List of Parameter nodes that should be trained
    """
    trainable = []
    seen = set()

    for node in graph:
        if type(node) == Parameter and id(node) not in seen:
            trainable.append(node)
            seen.add(id(node))

    return trainable
