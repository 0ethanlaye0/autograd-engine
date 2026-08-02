import random
import engine
from engine import Value

class Neuron():
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        out = sum(wi*xi for wi,xi in zip(self.w, x)) + self.b
        out = out.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin: {self.nin}, w: {self.w})"



class Layer():
    def __init__(self, nin, nout):
        self.nin = nin
        self.neurons = [Neuron(nin) for i in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        out = ""
        for n in self.neurons:
            out += f"Neuron(nin: {self.nin}, w: {n.w})\n"
        out += "\n\n"
        return out



class MLP():
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self):
        out = ""
        for l in self.layers:
            out += str(l)
        return out

#Model and data initialization

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]
mlp = MLP(2, [4, 4, 1])



#Training model
epochs = 1000
lr = 0.01
for epoch in range(epochs):
    preds = [mlp(x) for x in xs]
    loss = sum([(y-pred)**2 for y, pred in zip(ys,preds)])
    for p in mlp.parameters():
        p.grad = 0
    loss.backward()
    for p in mlp.parameters():
        p.data += -lr * p.grad
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | loss: {loss.data:.4f}")




