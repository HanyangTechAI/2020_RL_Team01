from .game import Game
from .agent import Agent
from random import choice


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def next(self, game: Game):
        print(str(game.state))
        return choice(game.get_valid_actions())


if __name__ == '__main__':
    game = Game()
    agent = RandomAgent()
    game.run(agent)
    print(game)

