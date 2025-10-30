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
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
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

class Linear(Node):
    """Linear computation node: y = Ax + b"""

    def __init__(self, x, n_output, n_input, initialization='He'):
        """
        Initialize Linear node with automatic parameter creation
        """

        # Initialize weights
        if initialization == 'he':
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

        # Initialize parent Node with all inputs
        Node.__init__(self, [x, self.A, self.b])


    def forward(self):
        """Compute forward pass: y = Ax + b"""
        x, A, b = self.inputs
        self.value = np.dot(A.value, x.value) + b.value

    def backward(self):
        """Compute gradients"""
        x, A, b = self.inputs
        grad_output = self.outputs[0].gradients[self]

        # dy/dx = A^T @ grad_output
        self.gradients[x] = np.dot(A.value.T, grad_output)

        # dy/dA = grad_output @ x^T
        self.gradients[A] = np.dot(grad_output, x.value.T)

        # dy/db = sum over batch dimension (keepdims for consistency)
        self.gradients[b] = np.sum(grad_output, axis=1, keepdims=True)


def topological_sort(output_node):
    """
    Perform topological sort on computational graph using DFS

    Args:
        output_node: The final output node of the graph (typically loss)

    Returns:
        List of nodes in topological order (suitable for forward pass)
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
