import numpy as np
import pandas as pd
import torch

from random_agent import Hai
from dqn_agent import DQNAgent


class Othello:
    # White: 1, Black: -1

    board = np.zeros((8, 8))
    board[3, 3], board[3, 4], board[4, 3], board[4, 4] = "1", "-1", "-1", "1"

    def __init__(self):

        super(Othello, self).__init__()

        self.board = Othello.board
        self.turn = "Black"
        self.pass_turn = 0
        self.game_end = False
        self.mode = 0
        self.player = 0

        self.black = 0  ## 대전 참고
        self.white = 0
        self.draw = 0

    def select_mode(self):
        print("Select Mode")
        print("1: Human vs HAI")
        print("2: HAI vs HAI")
        print("3: Human vs DQN_Agent")
        print("4: HAI(White) vs DQN_Agent(Black)")
        print("5: HAI(Black) vs DQN_Agent(White)")

        mode_1 = int(input())

        if mode_1 in [1, 3]:

            self.mode = mode_1
            print("\nSelect Your Color\n\n1: WHITE\t2: BLACK\n(Black Drops First)")
            mode_2 = int(input())

            if mode_2 == 1:
                self.player = 1
            elif mode_2 == 2:
                self.player = 2
            else:
                print("Wrong Input")

        elif mode_1 in [2, 4, 5]:

            self.mode = mode_1

        else:
            print("Wrong Input")

    def get_avail_pos(self):

        board = self.board
        # turn = 1 if self.turn == 'White' else -1
        if self.turn == "White":
            turn = 1
            oppose = -1
        else:
            turn = -1
            oppose = 1

        pos = [[i, j] for i in range(8) for j in range(8)]

        stone_exist = np.where(board != 0)

        exist_list = []
        for i in range(len(stone_exist[0])):
            exist_list.append([stone_exist[0][i], stone_exist[1][i]])

        for i in exist_list:
            pos.remove(i)

        pos_list = pos.copy()

        # up, down, left, right, ...
        directions = [
            [0, 1],
            [0, -1],
            [-1, 0],
            [1, 0],
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1],
        ]

        for curr_pos in pos_list:

            is_avail = False

            for x_dir, y_dir in directions:
                # new_pos = [x + y for x, y in zip(curr_pos, dir)]
                # new_pos = curr_pos + dir
                new_pos = curr_pos.copy()
                new_pos[0] += x_dir
                new_pos[1] += y_dir

                if new_pos[0] > 7 or new_pos[1] > 7 or new_pos[0] < 0 or new_pos[1] < 0:
                    continue

                elif (
                    self.board[new_pos[0], new_pos[1]] == turn
                    or self.board[new_pos[0], new_pos[1]] == 0
                ):
                    continue
                # if self.board[new_pos[0],new_pos[1]] != oppose: continue

                while True:

                    new_pos[0] += x_dir
                    new_pos[1] += y_dir

                    if (
                        new_pos[0] > 7
                        or new_pos[1] > 7
                        or new_pos[0] < 0
                        or new_pos[1] < 0
                    ):
                        break

                    elif self.board[new_pos[0], new_pos[1]] == 0:
                        break

                    elif self.board[new_pos[0], new_pos[1]] == turn:
                        is_avail = True
                        break

                    # new_pos += dir
                    # new_pos = [x + y for x, y in zip(new_pos, dir)]

                if is_avail:
                    break
                # else: continue

            if not is_avail:
                pos.remove(curr_pos)

        return pos  # 수정

    def flip_stones(self, curr_pos):

        board = self.board

        if self.turn == "White":
            turn = 1
            oppose = -1
        else:
            turn = -1
            oppose = 1

        directions = [
            [0, 1],
            [0, -1],
            [-1, 0],
            [1, 0],
            [1, 1],
            [1, -1],
            [-1, -1],
            [-1, 1],
        ]

        for x_dir, y_dir in directions:
            # new_pos = curr_pos + dir
            # new_pos = [x + y for x, y in zip(curr_pos, dir)]
            new_pos = curr_pos.copy()
            new_pos[0] += x_dir
            new_pos[1] += y_dir

            if new_pos[0] > 7 or new_pos[1] > 7 or new_pos[0] < 0 or new_pos[1] < 0:
                continue

            elif (
                self.board[new_pos[0], new_pos[1]] == turn
                or self.board[new_pos[0], new_pos[1]] == 0
            ):
                continue

            # if self.board[new_pos[0],new_pos[1]] != oppose: continue

            temporary_board = (self.board).copy()

            avail_flip = False

            while True:

                temporary_board[new_pos[0], new_pos[1]] = turn

                new_pos[0] += x_dir
                new_pos[1] += y_dir

                if new_pos[0] > 7 or new_pos[1] > 7 or new_pos[0] < 0 or new_pos[1] < 0:
                    break

                elif self.board[new_pos[0], new_pos[1]] == 0:
                    break

                elif self.board[new_pos[0], new_pos[1]] == turn:
                    avail_flip = True
                    break

                # new_pos = [x + y for x, y in zip(new_pos, dir)]
                # new_pos += dir

            # print(x_dir, y_dir, temporary_board)

            if avail_flip:
                self.board = temporary_board

    def player_input(self, avail_list):

        # avail_list = self.get_avail_pos()
        print(
            "\nAvailable Positions: ",
            avail_list,
            "\n\nDrop the stone ({}'s turn)\t(e.g. 0 1):".format(self.turn),
        )
        loc = list(map(int, input().split()))

        while True:

            if loc in avail_list:
                break

            print("\nThis position is not available.\n\nPlease Drop Again: ")

            loc = list(map(int, input().split()))

        return loc

    def drop_stone(self, loc):

        # # avail_list = self.get_avail_pos()
        # print('\nAvailable Positions: ', avail_list, '\n\nDrop the stone ({}\'s turn)\t(e.g. 0 1):'.format(self.turn))
        # loc = list(map(int, input().split()))
        #
        # while True:
        #
        #     if loc in avail_list:
        #         break
        #
        #     print('\nThis position is not available.\n\nPlease Drop Again: ')
        #
        #     loc = list(map(int, input().split()))

        self.flip_stones(curr_pos=loc)

        if self.turn == "White":
            self.board[loc[0], loc[1]] = 1
            self.turn = "Black"

        else:
            self.board[loc[0], loc[1]] = -1
            self.turn = "White"

        self.pass_turn = 0

    def no_place(self, avail_list):

        if len(avail_list) == 0:
            self.pass_turn += 1

            if self.turn == "White":
                self.turn = "Black"
            else:
                self.turn = "White"

            # self.is_end()

            return True

    def is_end(self):

        if (
            np.all(abs(self.board) > 0)
            or np.all(self.board != 1)
            or np.all(self.board != -1)
            or self.pass_turn == 2
        ):
            self.game_end = True
            print("\n\n\nThe End\n\n")
            if np.sum(self.board == 1) > np.sum(self.board == -1):
                print("★★★ Winner is {} ★★★".format("WHITE (●)"))
                self.white += 1
            elif np.sum(self.board == 1) < np.sum(self.board == -1):
                print("★★★ Winner is {} ★★★".format("Black (○)"))
                self.black += 1
            else:
                print("★★★ Draw ★★★")
                self.draw += 1

            self.show_board()
            print(
                "\nWhite: {} / Black: {}".format(
                    np.sum(self.board == 1), np.sum(self.board == -1)
                )
            )

    def is_end_train(self, color):

        if (
            np.all(abs(self.board) > 0)
            or np.all(self.board != 1)
            or np.all(self.board != -1)
            or self.pass_turn == 2
        ):
            self.game_end = True

            if np.sum(self.board == color) > np.sum(self.board == -color):
                return 1
            elif np.sum(self.board == color) < np.sum(self.board == -color):
                return -1
            else:
                return 0.5
        else:
            return 0

    def show_board(self):

        board = pd.DataFrame(self.board)
        board = board.applymap(lambda x: "●" if x == 1 else x)
        board = board.applymap(lambda x: "○" if x == -1 else x)
        board = board.applymap(lambda x: "-" if x == 0 else x)

        print("\n", board)

    def game_run(self):

        if self.mode == 1:
            agent = Hai()
        elif self.mode == 2:
            agent1 = Hai()  # White
            agent2 = Hai()  # Black
        elif self.mode == 3:
            agent = DQNAgent()
            if self.player == 1:
                agent.Q.load_state_dict(torch.load("model_black_agent.pth"))
            elif self.player == 2:
                agent.Q.load_state_dict(torch.load("model_white_agent.pth"))
        elif self.mode == 4:
            # if np.random.uniform() < 0.5:
            agent1 = Hai()  # White
            agent2 = DQNAgent()  # Black
            agent2.Q.load_state_dict(torch.load("model_black_agent.pth"))
        elif self.mode == 5:
            agent1 = DQNAgent()  # White
            agent1.Q.load_state_dict(torch.load("model_white_agent.pth"))
            agent2 = Hai()  # Black

        while not self.game_end:

            self.show_board()

            avail_pos = self.get_avail_pos()

            if self.no_place(avail_pos):
                print("\nThers's no place you can drop\n\nChange Turn")
                self.is_end()
                continue

            if self.mode == 1:  # Player vs HAI

                if self.player == 1:

                    if self.turn == "White":

                        loc = self.player_input(avail_pos)
                        self.drop_stone(loc)

                    elif self.turn == "Black":

                        loc = agent.input_avail_base(avail_pos)
                        self.drop_stone(loc)

                        print("\n\nAgent(" "Black" ") Drops")

                elif self.player == 2:

                    if self.turn == "White":

                        loc = agent.input_avail_base(avail_pos)
                        self.drop_stone(loc)

                        print("\n\nAgent(" "White" ") Drops")

                    elif self.turn == "Black":

                        loc = self.player_input(avail_pos)
                        self.drop_stone(loc)

            elif self.mode == 2:  # HAI vs HAI

                if self.turn == "White":

                    loc = agent1.input_no_base(avail_pos)
                    self.drop_stone(loc)

                    print("\n\nAgent(" "White" ") Drops")

                elif self.turn == "Black":

                    loc = agent2.input_no_base(avail_pos)
                    self.drop_stone(loc)

                    print("\n\nAgent(" "Black" ") Drops")

            elif self.mode == 3:  # Player vs DQN_Agent

                if self.player == 1:

                    if self.turn == "White":

                        loc = self.player_input(avail_pos)
                        self.drop_stone(loc)

                    elif self.turn == "Black":

                        s = self.board.flatten()
                        loc = agent.predict(s, avail_pos)
                        self.drop_stone(loc)

                        print("\n\nAgent(" "Black" ") Drops")

                elif self.player == 2:

                    if self.turn == "White":

                        s = self.board.flatten()
                        loc = agent.predict(s, avail_pos)
                        self.drop_stone(loc)

                        print("\n\nAgent(" "White" ") Drops")

                    elif self.turn == "Black":

                        loc = self.player_input(avail_pos)
                        self.drop_stone(loc)

            elif self.mode == 4:  # HAI(White) vs DQN_Agent(Black)

                if self.turn == "White":

                    loc = agent1.input_no_base(avail_pos)
                    self.drop_stone(loc)

                    print("\n\nHAI(" "White" ") Drops")

                elif self.turn == "Black":

                    s = self.board.flatten()
                    loc = agent2.predict(s, avail_pos)
                    self.drop_stone(loc)

                    print("\n\nDQN_Agent(" "Black" ") Drops")

            elif self.mode == 5:  # HAI(Black) vs DQN_Agent(White)

                if self.turn == "White":

                    s = self.board.flatten()
                    loc = agent1.predict(s, avail_pos)
                    self.drop_stone(loc)

                    print("\n\nDQN_Agent(" "White" ") Drops")

                elif self.turn == "Black":

                    loc = agent2.input_no_base(avail_pos)
                    self.drop_stone(loc)

                    print("\n\nHAI(" "Black" ") Drops")

            # othello.drop_stone(avail_pos)

            self.is_end()

    def iter_game(self, num):
        for _ in range(num):
            self.game_run()
        print(
            "White: {} | Black: {} | Draw: {}".format(self.white, self.black, self.draw)
        )
