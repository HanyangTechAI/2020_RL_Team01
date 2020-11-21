from .scoreboard import Scoreboard, LowerSection, UpperSection
from .dices import Dices
from enum import Enum


class Game:
    class State(Enum):
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

    def __rolling(self, agent):
        self.state = Game.State.Rolling

        self.dices.reset()
        self.dices.roll_all()
        while self.dices.can_roll:
            choice = agent.next(self)
            assert isinstance(choice, list)
            if len(choice) == 0:
                break
            else:
                self.dices.roll(choice)

    def __choosing_section(self, agent):
        self.state = Game.State.Choosing

        dices_arr = self.dices.array
        choice = agent.next(self)
        assert isinstance(choice, LowerSection) or isinstance(choice, UpperSection)
        if choice == LowerSection.Yahtzee:
            result, self.joker_score = self.scoreboard.score_yahtzee(dices_arr)

            if result == Scoreboard.YahtzeeResult.ChooseInLower:
                self.state = Game.State.ChooseInLower
                choice = agent.next(self)
                assert isinstance(choice, LowerSection)
                self.scoreboard.choose_joker(choice, self.joker_score)

            elif result == Scoreboard.YahtzeeResult.ChooseInUpper:
                self.state = Game.State.ChooseInUpper
                choice = agent.next(self)
                assert isinstance(choice, UpperSection)
                self.scoreboard.choose_joker(choice, self.joker_score)

        else:
            self.scoreboard.score(choice, dices_arr)

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

    def run(self, agent):
        self.scoreboard.reset()

        while not self.scoreboard.is_fulfilled:
            self.__rolling(agent)
            self.__choosing_section(agent)

        self.state = Game.State.NotInGame
