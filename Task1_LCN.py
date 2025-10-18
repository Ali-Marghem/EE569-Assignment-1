import numpy as np
from EDF import Node


class Linear(Node):
    """Linear computation node: y = Ax + b"""

    def __init__(self, x, A, b):
        """Initialize Linear node"""
        Node.__init__(self, [x, A, b])

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

        # dy/db = sum over batch dimension
        self.gradients[b] = np.sum(grad_output, axis=1).flatten()
