import numpy as np
import torch
import torch.nn as nn
from .agent import Agent
from .game import Game
import tensorboardX

NUM_STATE = 14 + 5 + 1 + 1  # scoreboard + dice + dice_roll_cnt + state_num
NUM_ACTION = 32             # dice selection(32), scoreboard selection(max 14)

LR = 0.00003
EPISODE = 10000
BATCH_SIZE = 128
GAMMA = 0.7
TRANSITIONS_CAPACITY = 300
UPDATE_DELAY = 20

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('torch device: ' + str(device))

# writer
writer = tensorboardX.SummaryWriter()


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
            nn.Linear(8 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, NUM_ACTION)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x


class DQNAgent(Agent):
    def __init__(self):
        super().__init__()
        self.transitions = np.zeros((TRANSITIONS_CAPACITY, 2 * NUM_STATE + 2))
        self.transitions_index = 0
        self.learn_iter = 0

        self.Q, self.Q_ = Net().to(device), Net().to(device)

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=LR)
        self.criteria = nn.MSELoss().to(device)

    def choose_action(self, x, game: Game, color, epsilon=0.1):
        available_actions = np.array(game.get_valid_actions(), dtype='int32')

        if np.random.uniform() < epsilon:
            action = np.random.choice(available_actions, 1)[0]
        else:
            x = torch.tensor(x, dtype=torch.float).to(device)
            x = x.view(1, -1)
            actions_values = self.Q(x)[0]

            ava_actions = actions_values[available_actions].clone().detach()

            _, action_ind = torch.max(ava_actions, 0)
            action = available_actions[action_ind]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.transitions[self.transitions_index %
                         TRANSITIONS_CAPACITY] = transition
        self.transitions_index += 1

    def learn(self, episode):
        losses = []
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

            batch_s = torch.tensor(batch_s, dtype=torch.float).to(device)
            batch_s_ = torch.tensor(batch_s_, dtype=torch.float).to(device)
            batch_a = torch.tensor(batch_a, dtype=int).to(device)
            batch_r = torch.tensor(batch_r, dtype=torch.float).to(device)

            batch_y = self.Q(batch_s).gather(1, batch_a)
            batch_y_ = self.Q_(
                batch_s_).detach()
            batch_y_ = batch_r + GAMMA * torch.max(batch_y_, 1)[0].view(-1, 1)

            loss = self.criteria(batch_y, batch_y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        writer.add_scalar('loss', sum(losses)/len(losses), episode)

    def next(self):
        return 0


if __name__ == "__main__":
    agent = DQNAgent()
    scores = []

    for episode in range(EPISODE * 50):
        game = Game()
        game.reset()
        while True:
            s = np.array(game.get_state(), dtype='int32')
            a = agent.choose_action(s, game, 1)
            score_bef = game.scoreboard.total_score
            game.add(a)
            # r = game.scoreboard.total_score - score_bef
            r = game.scoreboard.total_score
            s_ = np.array(game.get_state(), dtype='int32')

            agent.store_transition(s, a, r, s_)

            if game.scoreboard.is_fulfilled:
                agent.learn(episode)
                print('Episode:{} | score:{}'.format(episode, game.scoreboard.total_score))
                scores.append(game.scoreboard.total_score)
                break

        if (episode + 1) % 100 == 0:
            torch.save(agent.Q.state_dict(), 'model.pth')

            writer.add_scalar('score', sum(scores)/len(scores), episode)
            scores = []
    writer.close()

