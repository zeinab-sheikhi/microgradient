import random

from micrograd.engine import Value
from typing import List


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_ins: List[int], nonlin=True):
        self.weight = [Value(data=random.uniform(-1, 1)) for _ in range(n_ins)]
        self.bias = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([wi * xi for xi, wi in zip(x, self.weight)], self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weight + [self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.weight)})"


class Layer(Module):
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in: int, n_outs: List[int]):
        sz = [n_in] + n_outs
        self.layers = [
            Layer(n_in=sz[i], n_out=sz[i + 1], nonlin=i != len(n_outs) - 1) 
            for i in range(sz)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
