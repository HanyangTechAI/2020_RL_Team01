from .scoreboard import UpperSection, LowerSection, number_to_category, is_yahtzee
from .game import Game
from .agent import StateCategorizedAgent
from random import randint, choice


class RandomAgent(StateCategorizedAgent):
    @staticmethod
    def __get_available_list_between(game: Game, a, b):
        available_list = []
        for i in range(a, b + 1):
            category = number_to_category(i)
            if game.scoreboard[category] is None:
                available_list.append(category)
        return available_list

    def __init__(self):
        super().__init__()

    def _next_rolling(self, game: Game):
        re_roll = randint(0, 1) == 0
        if re_roll:
            select_set = set()
            for i in range(randint(1, 20)):
                select_set.add(randint(0, len(game.dices) - 1))
            return list(select_set)
        else:
            return []

    def _next_choosing(self, game: Game):
        dices_arr = game.dices.array

        available_list = RandomAgent.__get_available_list_between(game,
                                                                  UpperSection.Aces.value,
                                                                  LowerSection.Yahtzee.value)

        # though Yahtzee category is filled, player can choose Yahtzee if the condition is met
        if LowerSection.Yahtzee not in available_list and is_yahtzee(dices_arr):
            available_list.append(LowerSection.Yahtzee)

        return choice(available_list)

    def _next_choose_in_lower(self, game: Game):
        available_list = RandomAgent.__get_available_list_between(game,
                                                                  LowerSection.ThreeOfAKind.value,
                                                                  LowerSection.Chance.value)
        return choice(available_list)

    def _next_choose_in_upper(self, game: Game):
        available_list = RandomAgent.__get_available_list_between(game,
                                                                  UpperSection.Aces.value,
                                                                  UpperSection.Sixes.value)
        return choice(available_list)


if __name__ == '__main__':
    game = Game()
    agent = RandomAgent()
    game.run(agent)
    print(game)

