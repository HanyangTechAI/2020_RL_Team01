from .scoreboard import Scoreboard, UpperSection, LowerSection, number_to_category
from .dices import Dices
from abc import *


class Game(metaclass=ABCMeta):
    def __init__(self):
        self._scoreboard = Scoreboard()
        self._dices = Dices()

    @abstractmethod
    def _rolling(self):
        pass

    @abstractmethod
    def _choosing_section(self):
        pass

    @abstractmethod
    def _print_result(self):
        pass

    def run(self):
        self._scoreboard.reset()
        while not self._scoreboard.is_fulfilled:
            self._dices.reset()
            self._rolling()
            self._choosing_section()
        self._print_result()
