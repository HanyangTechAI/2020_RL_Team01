import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from othello import *

BOARD_SIZE = 8
NUM_STATE = pow(BOARD_SIZE, 2)
NUM_ACTION = pow(BOARD_SIZE, 2) + 1

LR = 0.001
EPISODE = 10000
BATCH_SIZE = 32
GAMMA = 0.9
ALPHA = 0.8
TRANSITIONS_CAPACITY = 200
UPDATE_DELAY = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(NUM_STATE, 128),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(8 * 128, NUM_ACTION)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x


class DQNAgent():
    def __init__(self, color):
        self.transitions = np.zeros((TRANSITIONS_CAPACITY, 2 * NUM_STATE + 2))
        self.transitions_index = 0
        self.learn_iter = 0

        self.Q, self.Q_ = Net(), Net()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=LR)
        self.criteria = nn.MSELoss()

    def choose_action(self, x, game_state, color, Epsilon=0.1):
        if color == 1:
            avaliable_pos = game_state.get_valid_pos(
                game_state.black_chess, game_state.white_chess)
        elif color == -1:
            avaliable_pos = game_state.get_valid_pos(
                game_state.white_chess, game_state.black_chess)

        avaliable_pos = list(
            map(lambda a: game_state.board_size * a[0] + a[1], avaliable_pos))
        if len(avaliable_pos) == 0:
            return 64

        if np.random.uniform() < Epsilon:
            action = np.random.choice(avaliable_pos, 1)[0]
        else:
            x = torch.tensor(x, dtype=torch.float)
            x = x.view(1, -1)
            actions_values = self.Q(x)[0]

            ava_actions = actions_values[avaliable_pos].clone().detach()

            _, action_ind = torch.max(ava_actions, 0)
            action = avaliable_pos[action_ind]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.transitions[self.transitions_index %
                         TRANSITIONS_CAPACITY] = transition
        self.transitions_index += 1

    def learn(self, oppo_Q_):
        for step in range(10):
            if self.learn_iter % UPDATE_DELAY == 0:
                self.Q_.load_state_dict(self.Q.state_dict())
            self.learn_iter += 1

            sample_index = np.random.choice(TRANSITIONS_CAPACITY,
                                            BATCH_SIZE)
            batch_tran = self.transitions[sample_index, :]
            batch_s = batch_tran[:, :NUM_STATE]
            batch_a = batch_tran[:, NUM_STATE: NUM_STATE + 1]
            batch_r = batch_tran[:, NUM_STATE + 1: NUM_STATE + 2]
            batch_s_ = batch_tran[:, NUM_STATE + 2:]

            batch_s = torch.tensor(batch_s, dtype=torch.float)
            batch_s_ = torch.tensor(batch_s_, dtype=torch.float)
            batch_a = torch.tensor(batch_a, dtype=int)
            batch_r = torch.tensor(batch_r, dtype=torch.float)

            batch_y = self.Q(batch_s).gather(1,
                                             batch_a)
            batch_y_ = oppo_Q_(
                batch_s_).detach()
            batch_y_ = batch_r - GAMMA * torch.max(batch_y_, 1)[0].view(-1, 1)

            loss = self.criteria(batch_y, batch_y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    offensive = DQNAgent(1)
    defensive = DQNAgent(-1)

    for episode in range(EPISODE * 50):
        game = Game()
        round_ = 0
        while True:
            round_ += 1
            s = game.get_state()
            a = offensive.choose_action(s, game, 1)
            game.add(1, a)
            r = game.gameover() * 100.0
            s_ = game.get_state()

            offensive.store_transition(s, a, r, s_)

            if r != 0 or round_ > 100:
                offensive.learn(defensive.Q_)
                print('Episode:{} | Reward:{}'.format(episode, r))
                break

            s = game.get_state()
            a = defensive.choose_action(s, game, -1)
            game.add(-1, a)
            r = game.gameover() * 100.0
            s_ = game.get_state()

            defensive.store_transition(s, a, -r, s_)

            if r != 0:
                defensive.learn(offensive.Q_)
                print('Episode:{} | Reward:{}'.format(episode, r))
                break

        if (episode + 1) % 100 == 0:
            torch.save(offensive.Q.state_dict(), 'model_offensive.pth')
            torch.save(defensive.Q.state_dict(), 'model_defensive.pth')