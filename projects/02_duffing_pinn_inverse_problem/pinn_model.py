import torch
import torch.nn as nn

class DuffingPINN(nn.Module):

    def __init__(self, hidden_width=32, hidden_layers=3):
        super().__init__()

        layers = [nn.Linear(1,hidden_width), nn.Tanh()]

        for _ in range(hidden_layers-1):
            layers.extend([nn.Linear(hidden_width,hidden_width), nn.Tanh()])

        layers.append(nn.Linear(hidden_width,2))

        self.network = nn.Sequential(*layers)

    def forward(self,t):
        return self.network(t)


if __name__ == "__main__":

    device = torch.device("cuda")
    model = DuffingPINN()
    model = model.to(device)
    print(f"Model offloaded to {device}")

    t_test = torch.linspace(0.0,20.0, 10).view(-1,1).to(device)

    prediction = model(t_test)

    print(f"Device: {device}")
    print(f"Input shape: {t_test.shape}")
    print(f"Output shape: {prediction.shape}")  