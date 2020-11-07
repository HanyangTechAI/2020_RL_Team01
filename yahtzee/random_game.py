from .scoreboard import Scoreboard, UpperSection, LowerSection, number_to_category, is_yahtzee
from .game import Game
from random import randint, choice


class RandomGame(Game):
    def __init__(self):
        super().__init__()

    def __get_available_list_between(self, a, b):
        available_list = []
        for i in range(a, b + 1):
            category = number_to_category(i)
            if self._scoreboard[category] is None:
                available_list.append(category)
        return available_list

    def _rolling(self):
        self._dices.roll_all()
        while self._dices.can_roll:
            re_roll = randint(0, 1) == 0
            if re_roll:
                select_set = set()
                for i in range(randint(1, 20)):
                    select_set.add(randint(0, len(self._dices)-1))
                self._dices.roll(list(select_set))
            else:
                break

    def _choosing_section(self):
        dices_arr = self._dices.array

        available_list = self.__get_available_list_between(UpperSection.Aces.value,
                                                           LowerSection.Yahtzee.value)
        # though Yahtzee category is filled, player can choose Yahtzee if the condition is met
        if LowerSection.Yahtzee not in available_list and is_yahtzee(dices_arr):
            available_list.append(LowerSection.Yahtzee)

        select = choice(available_list)

        if select == LowerSection.Yahtzee:
            result, joker_score = self._scoreboard.score_yahtzee(dices_arr)

            if result == Scoreboard.YahtzeeResult.ChooseInLower:
                available_list = self.__get_available_list_between(LowerSection.ThreeOfAKind.value,
                                                                   LowerSection.Chance.value)
                self._scoreboard.choose_joker(choice(available_list), joker_score)

            elif result == Scoreboard.YahtzeeResult.ChooseInUpper:
                available_list = self.__get_available_list_between(UpperSection.Aces.value,
                                                                   UpperSection.Sixes.value)
                self._scoreboard.choose_joker(choice(available_list), joker_score)

        else:
            self._scoreboard.score(select, dices_arr)

    def _print_result(self):
        print('========= Upper Section =========')
        for category in UpperSection:
            print('{}: {}'.format(category.name, self._scoreboard[category]))
        print('upper section total: {}'.format(self._scoreboard.upper_section_total))

        print('\n========= Lower Section =========')
        for category in LowerSection:
            print('{}: {}'.format(category.name, self._scoreboard[category]))
        print('lower section total: {}'.format(self._scoreboard.lower_section_total))

        print('\ntotal: {}\n'.format(self._scoreboard.total_score))


if __name__ == '__main__':
    game = RandomGame()
    game.run()

