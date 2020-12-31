import numpy as np
import torch
import torch.nn as nn


BOARD_SIZE = 8
NUM_STATE = pow(BOARD_SIZE, 2)
NUM_ACTION = pow(BOARD_SIZE, 2) + 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Sequential(nn.Linear(NUM_STATE, 128), nn.LeakyReLU())

        self.conv1 = nn.Sequential(nn.Conv1d(1, 4, 3, 1, 1), nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv1d(4, 8, 3, 1, 1), nn.LeakyReLU())

        self.linear2 = nn.Sequential(nn.Linear(8 * 128, NUM_ACTION))

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x


class DQNAgent:
    def __init__(self):

        self.board_size = 8

        self.Q = Net()

    def predict(self, x, available_pos):

        available_pos = list(map(lambda a: BOARD_SIZE * a[0] + a[1], available_pos))
        # print(available_pos)

        x = torch.tensor(x, dtype=torch.float)
        x = x.view(1, -1)

        self.Q.eval()
        actions_values = self.Q(x)[0]
        ava_actions = actions_values[available_pos].clone().detach()

        _, action_ind = torch.max(ava_actions, 0)
        action = available_pos[action_ind]
        loc = [action // self.board_size, action % self.board_size]

        return loc
