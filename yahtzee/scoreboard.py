from enum import Enum, auto


class UpperSection(Enum):
    Aces = 1
    Twos = 2
    Threes = 3
    Fours = 4
    Fives = 5
    Sixes = 6


class LowerSection(Enum):
    ThreeOfAKind = auto()
    FourOfAKind = auto()
    FullHouse = auto()
    SmallStraight = auto()
    LargeStraight = auto()
    Chance = auto()
    Yahtzee = auto()
    YahtzeeBonus = auto()


def dices_to_cnt(dices):
    cnt = [0] * 7
    for i in range(5):
        cnt[dices[i]] += 1
    return cnt


def dices_to_straights(dices):
    dices = sorted(dices)
    max_straight = 1
    current_straight = 1
    for i in range(1, 5):
        if dices[i] == dices[i - 1] + 1:
            current_straight += 1
            max_straight = max(current_straight, max_straight)
        elif dices[i] == dices[i - 1]:
            continue
        else:
            current_straight = 1
    return max_straight


def is_yahtzee(dices):
    cnt = dices_to_cnt(dices)
    return max(cnt) == len(dices)


class Scoreboard:
    def __init__(self):
        self.__upper_section = {}
        self.__lower_section = {}
        self.__joker_score = -1
        self.reset()

    def reset(self):
        self.__upper_section = {}
        for category in UpperSection:
            self.__upper_section[category] = -1

        self.__lower_section = {}
        for category in LowerSection:
            self.__lower_section[category] = -1
        self.__lower_section[LowerSection.YahtzeeBonus] = 0

        self.__joker_score = -1

    def __scoring_upper_section(self, category, dices):
        assert isinstance(category, UpperSection)
        assert self.__upper_section[category] == -1

        score = 0
        value = int(category.value)
        for n in dices:
            score += value if n == value else 0
        self.__upper_section[category] = score

    @staticmethod
    def __have_empty_category(section):
        for v in section.values():
            if v == -1:
                return True
        return False

    def is_empty(self, category):
        if isinstance(category, UpperSection):
            return self.__upper_section[category] == -1
        else:
            return self.__lower_section[category] == -1

    def get_score(self, category):
        if isinstance(category, UpperSection):
            return self.__upper_section[category]
        else:
            return self.__lower_section[category]

    def score(self, category, dices):
        assert not self.have_to_choose_joker
        assert self.is_empty(category)
        assert len(dices) == 5

        if isinstance(category, UpperSection):
            self.__scoring_upper_section(category, dices)
        else:
            if category == LowerSection.ThreeOfAKind:
                cnt = dices_to_cnt(dices)
                score = sum(dices) if max(cnt) >= 3 else 0
                self.__lower_section[LowerSection.ThreeOfAKind] = score

            if category == LowerSection.FourOfAKind:
                cnt = dices_to_cnt(dices)
                score = sum(dices) if max(cnt) >= 4 else 0
                self.__lower_section[LowerSection.FourOfAKind] = score

            if category == LowerSection.FullHouse:
                cnt = sorted(dices_to_cnt(dices))
                score = 25 if cnt[-1] == 3 and cnt[-2] == 2 else 0
                self.__lower_section[LowerSection.FullHouse] = score

            if category == LowerSection.SmallStraight:
                score = 30 if dices_to_straights(dices) >= 4 else 0
                self.__lower_section[LowerSection.SmallStraight] = score

            if category == LowerSection.LargeStraight:
                score = 40 if dices_to_straights(dices) >= 5 else 0
                self.__lower_section[LowerSection.LargeStraight] = score

            if category == LowerSection.Chance:
                score = sum(dices)
                self.__lower_section[LowerSection.Chance] = score

            if category == LowerSection.Yahtzee:
                assert False, "Yahtzee should be scored with 'yahtzee(dices)' method"

    class YahtzeeResult(Enum):
        ScoringSuccess = 0
        ChooseJokerInLower = 1
        ChooseJokerInUpper = 2

    def yahtzee(self, dices):
        assert not self.have_to_choose_joker
        assert len(dices) == 5

        if self.__lower_section[LowerSection.Yahtzee] == -1:
            self.__lower_section[LowerSection.Yahtzee] = 50 if is_yahtzee(dices) else 0
            return Scoreboard.YahtzeeResult.ScoringSuccess

        assert is_yahtzee(dices)

        if self.__lower_section[LowerSection.Yahtzee] != 0:
            self.__lower_section[LowerSection.YahtzeeBonus] += 100

        # joker rule
        alt_category = UpperSection(dices[0])
        score = dices[0]*len(dices)
        if self.__upper_section[alt_category] == -1:
            self.__upper_section[alt_category] = score
            return Scoreboard.YahtzeeResult.ScoringSuccess
        elif Scoreboard.__have_empty_category(self.__lower_section):
            self.__joker_score = score
            return Scoreboard.YahtzeeResult.ChooseJokerInLower
        else:
            self.__joker_score = 0
            return self.YahtzeeResult.ChooseJokerInUpper

    def choose_joker(self, category):
        assert self.have_to_choose_joker
        assert self.is_empty(category)

        if isinstance(category, UpperSection):
            assert self.__joker_score == 0
            self.__upper_section[category] = 0
            self.__joker_score = -1
        else:
            assert self.__joker_score > 0
            self.__lower_section[category] = self.__joker_score
            self.__joker_score = -1

    @property
    def have_to_choose_joker(self):
        return self.__joker_score != -1

    @property
    def total_score(self):
        upper_total = sum([v for v in self.__upper_section.values() if v != -1])
        total = upper_total + sum([v for v in self.__lower_section.values() if v != -1])
        total += 35 if upper_total >= 63 else 0
        return total

    @property
    def fulfilled(self):
        have_empty = Scoreboard.__have_empty_category(self.__upper_section) or \
                     Scoreboard.__have_empty_category(self.__lower_section)
        return not have_empty




