"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
from minitorch.tensor_functions import (
    View,
    tensor,
    Mul,
    Sum,
    Add
)

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
# TODO: Implement for Task 2.5.

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        x = self.layer3.forward(x).sigmoid()
        return x


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, x):
        """
        Forward pass for the Linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 2).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_size).
        """
        # Reshape x to (batch_size, in_size, 1)
        x_reshaped = View.apply(x, tensor([x.shape[0], x.shape[1]]))

        x_reshaped = View.apply(x, tensor([x_reshaped.shape[0], x_reshaped.shape[1], 1]))

        # Reshape weight to (1, in_size, out_size)
        weight_reshaped = View.apply(self.weight.value, tensor([1, self.weight.value.shape[0], self.weight.value.shape[1]]))

        # Element-wise multiplication
        prod = Mul.apply(x_reshaped, weight_reshaped)

        # Sum along the in_size dimension
        result = Sum.apply(prod, tensor([1]))

        # Add bias
        result = Add.apply(result, self.bias.value)
        result = View.apply(result, tensor([x_reshaped.shape[0], self.weight.value.shape[1]]))

        return result



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            epsilon = 1e-7
            loss = -(prob+epsilon).log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
