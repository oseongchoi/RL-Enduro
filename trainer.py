import torch
from torch import nn, optim


class Trainer:
    """
    Double DQN training implementation.
    """
    def __init__(self, online, target, gamma):
        """
        Initialize the instance.
        """
        self.online = online
        self.target = target
        self.update()
        self.optimizer = optim.RMSprop(
            self.online.parameters(), lr=0.0003, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train(self, state, action, reward, consequence, done):
        """
        Train the network on the given mini-batch.
        """
        state = torch.as_tensor(state, device='cuda')
        consequence = torch.as_tensor(consequence, device='cuda')

        self.online.train()
        self.optimizer.zero_grad()
        pred = self.online(state)
        true = pred.clone()

        self.target.eval()
        with torch.no_grad():
            value = self.target(consequence)

        for i, a in enumerate(action):
            if done[i]:
                true[i, a] = reward[i]
            else:
                true[i, a] = reward[i] + self.gamma * max(value[i])

        loss = self.criterion(pred, true)
        loss.backward()
        self.optimizer.step()

    def update(self):
        """
        Update target network.
        """
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        """
        Save the model as the given name.
        """
        torch.save(self.online.state_dict(), path)
