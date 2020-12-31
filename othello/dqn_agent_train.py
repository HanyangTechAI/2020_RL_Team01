import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from othello import Othello


BOARD_SIZE = 8
NUM_STATE = pow(BOARD_SIZE, 2)
NUM_ACTION = pow(BOARD_SIZE, 2) + 1

LR = 1e-10
EPISODE = 10000
BATCH_SIZE = 128
GAMMA = 0.9
ALPHA = 0.8
TRANSITIONS_CAPACITY = 200
UPDATE_DELAY = 10


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.linear1 = nn.Sequential(
#             nn.Linear(NUM_STATE, 128),
#             nn.LeakyReLU()
#         )
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(1, 4, 3, 1, 1),
#             nn.LeakyReLU(inplace=True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(4, 8, 3, 1, 1),
#             nn.LeakyReLU()
#         )
#
#         self.linear2 = nn.Sequential(
#             nn.Linear(8 * 128, NUM_ACTION)
#         )
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = x.view(x.shape[0], 1, -1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.shape[0], -1)
#         x = self.linear2(x)
#         return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(20736, 10000)
        self.fc2 = nn.Linear(10000, NUM_ACTION)

    def forward(self, x):

        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 20736)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))

        return x


class DQNAgent:
    def __init__(self):
        self.transitions = np.zeros((TRANSITIONS_CAPACITY, 2 * NUM_STATE + 2))
        self.transitions_index = 0
        self.learn_iter = 0
        self.board_size = 8

        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.3

        self.Q, self.Q_ = Net(), Net()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=LR)
        self.criteria = nn.MSELoss()

    def get_epsilon(self, total_episode, cur_episode):

        epsilon = self.epsilon * (
            1 - cur_episode / (total_episode * self.epsilon_decay)
        )

        if epsilon < self.epsilon_min:
            epsilon = self.epsilon_min

        return epsilon

    def choose_action(self, x, available_pos, total, cur):
        # if color == 1:
        #     avaliable_pos = game_state.get_valid_pos(
        #         game_state.black_chess, game_state.white_chess)
        # elif color == -1:
        #     avaliable_pos = game_state.get_valid_pos(
        #         game_state.white_chess, game_state.black_chess)

        available_pos = list(map(lambda a: BOARD_SIZE * a[0] + a[1], available_pos))
        # print(available_pos)

        if len(available_pos) == 0:
            return 64

        if np.random.uniform() < self.get_epsilon(total, cur):
            action = np.random.choice(available_pos, 1)[0]
            # print('!!!!!!!!!!!!!!!random!!!!!!!!!!!')
        else:
            x = torch.tensor(x, dtype=torch.float)
            x = x.view(1, -1)
            actions_values = self.Q(x)[0]
            ava_actions = actions_values[available_pos].clone().detach()

            _, action_ind = torch.max(ava_actions, 0)
            action = available_pos[action_ind]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.transitions[self.transitions_index % TRANSITIONS_CAPACITY] = transition
        self.transitions_index += 1

    def learn(self, oppo_Q_, episode):

        total_loss = 0

        for step in range(10):
            if self.learn_iter % UPDATE_DELAY == 0:
                self.Q_.load_state_dict(self.Q.state_dict())
            self.learn_iter += 1

            sample_index = np.random.choice(TRANSITIONS_CAPACITY, BATCH_SIZE)
            batch_tran = self.transitions[sample_index, :]
            batch_s = batch_tran[:, :NUM_STATE]
            batch_a = batch_tran[:, NUM_STATE : NUM_STATE + 1]
            batch_r = batch_tran[:, NUM_STATE + 1 : NUM_STATE + 2]
            batch_s_ = batch_tran[:, NUM_STATE + 2 :]

            batch_s = torch.tensor(batch_s, dtype=torch.float)
            batch_s_ = torch.tensor(batch_s_, dtype=torch.float)
            batch_a = torch.tensor(batch_a, dtype=torch.int64)
            batch_r = torch.tensor(batch_r, dtype=torch.float)

            batch_y = self.Q(batch_s).gather(1, batch_a)  ### 어떤 형태로?
            batch_y_ = oppo_Q_(batch_s_).detach()
            batch_y_ = batch_r - GAMMA * torch.max(batch_y_, 1)[0].view(-1, 1)

            loss = self.criteria(batch_y, batch_y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        print("Loss: ", total_loss / 10)

    # def predict(self, x, available_pos):
    #
    #     available_pos = list(
    #         map(lambda a: BOARD_SIZE * a[0] + a[1], available_pos))
    #     # print(available_pos)
    #
    #     x = torch.tensor(x, dtype=torch.float)
    #     x = x.view(1, -1)
    #
    #     self.Q.eval()
    #     actions_values = self.Q(x)[0]
    #     ava_actions = actions_values[available_pos].clone().detach()
    #
    #     _, action_ind = torch.max(ava_actions, 0)
    #     action = available_pos[action_ind]
    #     loc = [action // self.board_size, action % self.board_size]
    #
    #     return loc


if __name__ == "__main__":
    black_agent = DQNAgent()  # Black: -1 (color)
    white_agent = DQNAgent()  # White: 1 (color)

    TOTAL_EPISODE = EPISODE * 50

    for episode in range(TOTAL_EPISODE):

        othello = Othello()

        round_ = 0
        while True:
            round_ += 1

            avail_pos = othello.get_avail_pos()
            zero_avail = othello.no_place(avail_pos)

            if zero_avail:
                s = othello.board.flatten()
                a = black_agent.choose_action(s, avail_pos, TOTAL_EPISODE, episode)

            else:
                s = othello.board.flatten()
                a = black_agent.choose_action(s, avail_pos, TOTAL_EPISODE, episode)
                othello.drop_stone([a // BOARD_SIZE, a % BOARD_SIZE])

            r = othello.is_end_train(color=-1) * 100.0
            s_ = othello.board.flatten()

            black_agent.store_transition(s, a, r, s_)

            if othello.game_end or round_ > 100:
                black_agent.learn(white_agent.Q_, episode)
                print("Episode:{} | Reward:{} | Last:{}".format(episode, r, "Black"))
                print(
                    "Winner: {} | Round: {}".format(
                        "Black" if r == 100 else "White" if r == -100 else "Draw",
                        round_,
                    )
                )
                print("")
                break

            avail_pos = othello.get_avail_pos()
            zero_avail = othello.no_place(avail_pos)

            if zero_avail:
                s = othello.board.flatten()
                a = white_agent.choose_action(s, avail_pos, TOTAL_EPISODE, episode)

            else:
                s = othello.board.flatten()
                a = white_agent.choose_action(s, avail_pos, TOTAL_EPISODE, episode)
                othello.drop_stone([a // BOARD_SIZE, a % BOARD_SIZE])

            r = othello.is_end_train(color=1) * 100.0
            s_ = othello.board.flatten()

            white_agent.store_transition(s, a, r, s_)

            if othello.game_end:
                white_agent.learn(black_agent.Q_, episode)
                print("Episode:{} | Reward:{} | Last:{}".format(episode, r, "White"))
                print(
                    "Winner: {} | Round: {}".format(
                        "White" if r == 100 else "Black" if r == -100 else "Draw",
                        round_,
                    )
                )
                print("")
                break

        if (episode + 1) % 100 == 0:
            torch.save(black_agent.Q.state_dict(), "model_black_agent.pth")
            torch.save(white_agent.Q.state_dict(), "model_white_agent.pth")
            print(
                "\nEpisode: {} | Current_Epsilon: {}\n".format(
                    episode, black_agent.get_epsilon(TOTAL_EPISODE, episode)
                )
            )
        if (episode + 1) % 20000 == 0:
            torch.save(
                black_agent.Q.state_dict(),
                "model_black_agent_{}.pth".format(episode + 1),
            )
            torch.save(
                white_agent.Q.state_dict(),
                "model_white_agent_{}.pth".format(episode + 1),
            )
