from .scoreboard import UpperSection, LowerSection, number_to_category
from .game import Game
from .agent import StateCategorizedAgent
import os
import platform


class ConsoleAgent(StateCategorizedAgent):
    @staticmethod
    def __clear_console():
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    @staticmethod
    def __select_category_between(game: Game, a, b):
        select = -1
        while not a <= select <= b:
            select = int(input('어디에 채우시겠습니까? ({}~{})'.format(a, b)))
            if game.scoreboard[number_to_category(select)] is not None:
                select = 0
                print('빈 곳을 입력해주세요')
        return number_to_category(select)

    def __init__(self):
        super().__init__()

    def _next_rolling(self, game: Game):
        dices_arr = game.dices.array

        ConsoleAgent.__clear_console()
        print(game)
        print('현재 주사위')
        print(' '.join(map(str, dices_arr)))

        re_roll = None
        while re_roll != 'n' and re_roll != 'y':
            re_roll = input('다시 굴리겠습니까? (Y/N)').lower()
            if re_roll == 'n':
                return []
            elif re_roll == 'y':
                print('어떤 주사위들을 다시 굴리시겠습니까? (1~5, 여러개 입력 가능)')
                select = set(list(map(int, input().split())))
                select = [v - 1 for v in select if 1 <= v <= 5]
                return select
            else:
                print('(Y/N) 중 하나를 입력해주세요.')

    def _next_choosing(self, game: Game):
        dices_arr = game.dices.array
        ConsoleAgent.__clear_console()
        print(game)
        print('현재 주사위')
        print(' '.join(map(str, dices_arr)))

        select = ConsoleAgent.__select_category_between(game,
                                                        UpperSection.Aces.value,
                                                        LowerSection.Yahtzee.value)
        return select

    def _next_choose_in_lower(self, game: Game):
        select = ConsoleAgent.__select_category_between(game,
                                                        LowerSection.ThreeOfAKind.value,
                                                        LowerSection.Chance.value)
        return select

    def _next_choose_in_upper(self, game: Game):
        select = ConsoleAgent.__select_category_between(game,
                                                        UpperSection.Aces.value,
                                                        UpperSection.Sixes.value)
        return select


if __name__ == '__main__':
    game = Game()
    agent = ConsoleAgent()
    game.run(agent)
    print(game)

