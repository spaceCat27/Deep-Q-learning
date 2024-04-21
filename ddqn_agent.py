from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import random



class NeuralNetwork(nn.Module):
    def __init__(self, action_n, input_n, neuron_n) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_n, neuron_n)
        self.l2 = nn.Linear(neuron_n, action_n)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        
        return x

class DDQNAgent(nn.Module):
    def __init__(self, lr, action_n, input_n, discount, epsilon_decay) -> None:
        super().__init__()
        self.action_n = action_n
        self.model = NeuralNetwork(action_n, input_n, 24)
        self.target_model = NeuralNetwork(action_n, input_n, 24)
        self.discount = discount
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2_000)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 32


    def action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_n - 1)
        else:
            with torch.no_grad():
                action = torch.argmax(self.model(torch.from_numpy(state))).item()

        return action
    
    def get_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, k=self.batch_size)

        for state, action, reward, next_state, done in batch:
        
            if done:
                y = reward
            else:
                with torch.no_grad():
                    y = reward + self.discount * torch.max(self.target_model(torch.from_numpy(next_state))).item()

            y_hat = self.model(torch.from_numpy(state))
            y_target = y_hat.detach().clone()
            y_target[action] = y 
            loss = F.mse_loss(y_hat, y_target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def epsilon_change(self):
        if self.epsilon > 0.02:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        self.target_model.load_state_dict(self.model.state_dict())
