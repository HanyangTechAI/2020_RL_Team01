from random import randrange


class Dices:
    def __init__(self):
        self.__rollCount = 0
        self.__dices = [0]*5

    def can_roll(self):
        return self.__rollCount < 3

    def have_rolled(self):
        return self.__rollCount > 0

    def roll_all(self):
        assert self.can_roll()

        self.__rollCount += 1
        for i in range(len(self.__dices)):
            self.__dices[i] = randrange(1, 6+1)

    def roll(self, select):
        assert self.__rollCount > 0
        assert self.can_roll()

        self.__rollCount += 1
        for i in select:
            self.__dices[i] = randrange(1, 6+1)

    @property
    def get(self):
        return self.__dices


