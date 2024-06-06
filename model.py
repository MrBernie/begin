import dependency as d

class Encoder(d.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = d.nn.Sequential(d.nn.Linear(28 * 28, 64), d.nn.ReLU(), d.nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(d.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = d.nn.Sequential(d.nn.Linear(3, 64), d.nn.ReLU(), d.nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    
