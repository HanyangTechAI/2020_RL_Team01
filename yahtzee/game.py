from .scoreboard import Scoreboard, LowerSection, UpperSection, number_to_category, is_yahtzee
from .dices import Dices
from enum import IntEnum


class Game:
    class State(IntEnum):
        NotInGame = 0,
        Rolling = 1,
        Choosing = 2,
        ChooseInLower = 3,
        ChooseInUpper = 4

    def __init__(self):
        self.scoreboard = Scoreboard()
        self.dices = Dices()
        self.state = Game.State.NotInGame
        self.joker_score = None

    def __get_available_list_between(self, a, b):
        available_list = []
        for i in range(a, b + 1):
            category = number_to_category(i)
            if self.scoreboard[category] is None:
                available_list.append(i)
        return available_list

    def get_valid_actions(self):
        if self.state == Game.State.Rolling:
            return list(range(32))
        if self.state == Game.State.Choosing:
            valid_actions = self.__get_available_list_between(UpperSection.Aces.value,
                                                              LowerSection.Yahtzee.value)
            # though Yahtzee category is filled, player can choose Yahtzee if the condition is met
            if LowerSection.Yahtzee not in valid_actions and is_yahtzee(self.dices.array):
                valid_actions.append(LowerSection.Yahtzee.value)
            return valid_actions
        elif self.state == Game.State.ChooseInLower:
            return self.__get_available_list_between(LowerSection.ThreeOfAKind.value,
                                                     LowerSection.Chance.value)
        elif self.state == Game.State.ChooseInUpper:
            return self.__get_available_list_between(UpperSection.Aces.value,
                                                     UpperSection.Sixes.value)

    def get_state(self):
        state = []

        for category in UpperSection:
            score = self.scoreboard[category]
            state.append(-1 if score is None else score)
        for category in LowerSection:
            score = self.scoreboard[category]
            state.append(-1 if score is None else score)

        dices = self.dices.array
        for dice in dices:
            state.append(dice)
        state.append(self.dices.roll_cnt)

        state.append(int(self.state))

        return state

    def __str__(self):
        result = ""

        result += '========= Upper Section =========\n'
        for category in UpperSection:
            score = self.scoreboard[category]
            value = '@{}'.format(category.value) if score is None else score
            result += '{}: {}\n'.format(category.name, value)
        result += 'upper section total: {}\n'.format(self.scoreboard.upper_section_total)

        result += '\n========= Lower Section =========\n'
        for category in LowerSection:
            score = self.scoreboard[category]
            value = '@{}'.format(category.value) if score is None else score
            result += '{}: {}\n'.format(category.name, value)
        result += 'lower section total: {}'.format(self.scoreboard.lower_section_total)

        result += '\ntotal: {}\n'.format(self.scoreboard.total_score)

        return result

    def reset(self):
        self.scoreboard.reset()
        self.dices.reset()
        self.state = Game.State.Rolling
        self.dices.roll_all()

    def __action_to_dice_choices(self, n):
        choices = []
        for i in range(len(self.dices)):
            if n % 2 == 1:
                choices.append(i)
            n = int(n/2)
        return choices

    def add(self, action):
        if self.state == Game.State.Rolling:
            choice = self.__action_to_dice_choices(action)
            assert isinstance(choice, list)
            if len(choice) == 0:
                self.state = Game.State.Choosing
                return
            else:
                self.dices.roll(choice)
                if not self.dices.can_roll:
                    self.state = Game.State.Choosing

        elif self.state == Game.State.Choosing:
            dices_arr = self.dices.array
            choice = number_to_category(action)
            assert isinstance(choice, LowerSection) or isinstance(choice, UpperSection)
            if choice == LowerSection.Yahtzee:
                result, self.joker_score = self.scoreboard.score_yahtzee(dices_arr)
                if result == Scoreboard.YahtzeeResult.ChooseInLower:
                    self.state = Game.State.ChooseInLower
                elif result == Scoreboard.YahtzeeResult.ChooseInUpper:
                    self.state = Game.State.ChooseInUpper
            else:
                self.scoreboard.score(choice, dices_arr)
                self.state = Game.State.Rolling
                self.dices.reset()

        elif self.state == Game.State.ChooseInLower:
            choice = number_to_category(action)
            assert isinstance(choice, LowerSection)
            self.scoreboard.choose_joker(choice, self.joker_score)
            self.state = Game.State.Rolling
            self.dices.reset()

        elif self.state == Game.State.ChooseInUpper:
            choice = number_to_category(action)
            assert isinstance(choice, UpperSection)
            self.scoreboard.choose_joker(choice, self.joker_score)
            self.state = Game.State.Rolling
            self.dices.reset()

        # if next state is Game.State.Rolling and dice is not rolled, roll dice
        if self.state == Game.State.Rolling and not self.dices.have_rolled:
            self.dices.reset()
            self.dices.roll_all()

    def run(self, agent):
        self.reset()

        while not self.scoreboard.is_fulfilled:
            action = agent.next(self)
            self.add(action)

        self.state = Game.State.NotInGame
