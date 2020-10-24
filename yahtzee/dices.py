from random import randrange


class Dices:
    max_num = 6
    dices_cnt = 5
    max_roll_cnt = 3

    def __init__(self):
        self.__rollCount = 0
        self.__dices = [0]*Dices.dices_cnt

    @property
    def can_roll(self):
        return self.__rollCount < Dices.max_roll_cnt

    @property
    def have_rolled(self):
        return self.__rollCount > 0

    def roll_all(self):
        assert self.can_roll

        self.__rollCount += 1
        for i in range(len(self.__dices)):
            self.__dices[i] = randrange(1, 6+1)

    def roll(self, select):
        assert self.have_rolled
        assert self.can_roll

        self.__rollCount += 1
        for i in select:
            self.__dices[i] = randrange(1, Dices.max_num+1)

    def reset(self):
        self.__rollCount = 0
        self.__dices = [0]*Dices.dices_cnt

    @property
    def array(self):
        return list(self.__dices)

    def __len__(self):
        return len(self.__dices)

    def __getitem__(self, key):
        return self.__dices[key]


