from .scoreboard import Scoreboard, UpperSection, LowerSection, number_to_category
from .game import Game
import os
import platform


def clear_console():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


class ConsoleGame(Game):
    def __init__(self):
        super().__init__()

    def __select_category_between(self, a, b):
        select = 0
        while not a <= select <= b:
            select = int(input('어디에 채우시겠습니까? ({}~{})'.format(a, b)))
            if self._scoreboard[number_to_category(select)] is not None:
                select = 0
                print('빈 곳을 입력해주세요')
        return select

    def __print_state(self):
        print('========= Upper Section =========')
        for category in UpperSection:
            score = self._scoreboard[category]
            value = '@{}'.format(category.value) if score is None else score
            print('{}: {}'.format(category.name, value))
        print('upper section total: {}'.format(self._scoreboard.upper_section_total))

        print('\n========= Lower Section =========')
        for category in LowerSection:
            score = self._scoreboard[category]
            value = '@{}'.format(category.value) if score is None else score
            print('{}: {}'.format(category.name, value))
        print('lower section total: {}'.format(self._scoreboard.lower_section_total))

        print('\ntotal: {}\n'.format(self._scoreboard.total_score))

    def _rolling(self):
        self._dices.roll_all()
        while self._dices.can_roll:
            dices_arr = self._dices.array

            clear_console()
            self.__print_state()
            print('현재 주사위')
            print(' '.join(map(str, dices_arr)))

            re_roll = None
            while re_roll != 'n' and re_roll != 'y':
                re_roll = input('다시 굴리겠습니까? (Y/N)').lower()
                if re_roll == 'n':
                    break
                elif re_roll == 'y':
                    print('어떤 주사위들을 다시 굴리시겠습니까? (1~5, 여러개 입력 가능)')
                    select = set(list(map(int, input().split())))
                    select = [v - 1 for v in select if 1 <= v <= 5]
                    self._dices.roll(select)
                else:
                    print('(Y/N) 중 하나를 입력해주세요.')
            if re_roll == 'n':
                break

    def _choosing_section(self):
        dices_arr = self._dices.array

        clear_console()
        self.__print_state()
        print('현재 주사위')
        print(' '.join(map(str, dices_arr)))

        select = self.__select_category_between(UpperSection.Aces.value,
                                                LowerSection.Yahtzee.value)

        if select == LowerSection.Yahtzee.value:
            result, joker_score = self._scoreboard.score_yahtzee(dices_arr)

            if result == Scoreboard.YahtzeeResult.ChooseInLower:
                select = self.__select_category_between(LowerSection.ThreeOfAKind.value,
                                                        LowerSection.Chance.value)
                self._scoreboard.choose_joker(LowerSection(select), joker_score)

            elif result == Scoreboard.YahtzeeResult.ChooseInUpper:
                select = self.__select_category_between(UpperSection.Aces.value,
                                                        UpperSection.Sixes.value)
                self._scoreboard.choose_joker(UpperSection(select), joker_score)

        else:
            self._scoreboard.score(number_to_category(select), dices_arr)

    def _print_result(self):
        self.__print_state()


if __name__ == '__main__':
    game = ConsoleGame()
    game.run()

