from .game import Game
from abc import *


class Agent(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def next(self, game: Game):
        pass


class StateCategorizedAgent(Agent):
    def next(self, game: Game):
        if game.state == Game.State.Rolling:
            return self._next_rolling(game)
        elif game.state == Game.State.Choosing:
            return self._next_choosing(game)
        elif game.state == Game.State.ChooseInLower:
            return self._next_choose_in_lower(game)
        elif game.state == Game.State.ChooseInUpper:
            return self._next_choose_in_upper(game)
        else:
            return None

    @abstractmethod
    def _next_rolling(self, game: Game):
        pass

    @abstractmethod
    def _next_choosing(self, game: Game):
        pass

    @abstractmethod
    def _next_choose_in_lower(self, game: Game):
        pass

    @abstractmethod
    def _next_choose_in_upper(self, game: Game):
        pass
