import math


class Value:
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(
            data=self.data + other.data,
            label='+',
            children=(self, other)
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(
            data=self.data * other.data,
            label='*',
            children=(self, other)
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float data types are supported!"
        out = Value(
            data=self.data ** other,
            label=f'**{other}',
            children=(self, )
        )

        def _backward():
            self.grad += out.grad * (other * (self.data) ** (other - 1))
        out._backward = _backward
        return out

    def relu(self):
        out = Value(
            data=self.data if self.data > 0 else 0,
            label='ReLU',
            children=(self, )
        )

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(
            data=math.exp(self.data),
            label='exp',
            children=(self, )
        )

        def _backward():
            self.grad += out.grad * out
        out._backward = _backward
        return out

    def tanh(self):
        exp_2n = (self * 2).exp()
        out = Value(
            data=(exp_2n - 1) / (exp_2n + 1),
            label='tanh',
            children=(self, )
        )

        def _backward():
            self.grad += out.grad * (1 - out ** 2)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
