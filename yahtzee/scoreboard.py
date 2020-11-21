from enum import Enum
from .dices import Dices


class UpperSection(Enum):
    Aces = 1
    Twos = 2
    Threes = 3
    Fours = 4
    Fives = 5
    Sixes = 6


class LowerSection(Enum):
    ThreeOfAKind = 7
    FourOfAKind = 8
    FullHouse = 9
    SmallStraight = 10
    LargeStraight = 11
    Chance = 12
    Yahtzee = 13
    YahtzeeBonus = 14


def number_to_category(n):
    if UpperSection.Aces.value <= n <= UpperSection.Sixes.value:
        return UpperSection(n)
    elif LowerSection.ThreeOfAKind.value <= n <= LowerSection.YahtzeeBonus.value:
        return LowerSection(n)
    else:
        assert False, "invalid number"


def dices_to_cnt(dices):
    cnt = [0] * (Dices.MAX_NUM + 1)
    for i in range(len(dices)):
        cnt[dices[i]] += 1
    return cnt


def dices_to_straights(dices):
    dices = sorted(dices)
    max_straight = 1
    current_straight = 1
    for i in range(1, len(dices)):
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


# None: score varies depending on the dices
SCORE_BY_CATEGORY = {
    UpperSection.Aces: None,
    UpperSection.Twos: None,
    UpperSection.Threes: None,
    UpperSection.Fours: None,
    UpperSection.Fives: None,
    UpperSection.Sixes: None,

    LowerSection.ThreeOfAKind: None,
    LowerSection.FourOfAKind: None,
    LowerSection.FullHouse: 25,
    LowerSection.SmallStraight: 30,
    LowerSection.LargeStraight: 40,
    LowerSection.Chance: None,
    LowerSection.Yahtzee: 50,
    LowerSection.YahtzeeBonus: 100
}

UPPER_SECTION_BONUS = 35
UPPER_SECTION_BONUS_CONDITION = 63


class Scoreboard:
    def __init__(self):
        self.__upper_section = {}
        self.__lower_section = {}
        self.reset()

    def reset(self):
        self.__upper_section = {}
        for category in UpperSection:
            self.__upper_section[category] = None

        self.__lower_section = {}
        for category in LowerSection:
            self.__lower_section[category] = None
        self.__lower_section[LowerSection.YahtzeeBonus] = 0

    def __score_upper_section(self, category, dices):
        assert isinstance(category, UpperSection)
        assert self.__upper_section[category] is None

        score = 0
        value = int(category.value)
        for n in dices:
            score += value if n == value else 0
        self.__upper_section[category] = score

    @staticmethod
    def __have_empty_category(section):
        for v in section.values():
            if v is None:
                return True
        return False

    def __getitem__(self, category):
        if isinstance(category, UpperSection):
            return self.__upper_section[category]
        else:
            return self.__lower_section[category]

    def score(self, category, dices):
        assert self[category] is None
        assert len(dices) == 5

        if isinstance(category, UpperSection):
            self.__score_upper_section(category, dices)
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
                score = SCORE_BY_CATEGORY[LowerSection.FullHouse] if cnt[-1] == 3 and cnt[-2] == 2 else 0
                self.__lower_section[LowerSection.FullHouse] = score

            if category == LowerSection.SmallStraight:
                score = SCORE_BY_CATEGORY[LowerSection.SmallStraight] if dices_to_straights(dices) >= 4 else 0
                self.__lower_section[LowerSection.SmallStraight] = score

            if category == LowerSection.LargeStraight:
                score = SCORE_BY_CATEGORY[LowerSection.LargeStraight] if dices_to_straights(dices) >= 5 else 0
                self.__lower_section[LowerSection.LargeStraight] = score

            if category == LowerSection.Chance:
                score = sum(dices)
                self.__lower_section[LowerSection.Chance] = score

            if category == LowerSection.Yahtzee:
                assert False, "Yahtzee should be scored with 'yahtzee(dices)' method"

    class YahtzeeResult(Enum):
        Scored = 0
        ChooseInLower = 1
        ChooseInUpper = 2

    # return (YahtzeeResult, alt_score)
    def score_yahtzee(self, dices):
        assert len(dices) == 5

        if self.__lower_section[LowerSection.Yahtzee] is None:
            self.__lower_section[LowerSection.Yahtzee] = SCORE_BY_CATEGORY[LowerSection.Yahtzee] if is_yahtzee(dices) else 0
            return Scoreboard.YahtzeeResult.Scored, None

        assert is_yahtzee(dices)

        if self.__lower_section[LowerSection.Yahtzee] != 0:
            self.__lower_section[LowerSection.YahtzeeBonus] += SCORE_BY_CATEGORY[LowerSection.YahtzeeBonus]

        # joker rule
        alt_category = UpperSection(dices[0])
        alt_score = dices[0]*len(dices)
        if self.__upper_section[alt_category] is None:
            self.__upper_section[alt_category] = alt_score
            return Scoreboard.YahtzeeResult.Scored, None
        elif Scoreboard.__have_empty_category(self.__lower_section):
            return Scoreboard.YahtzeeResult.ChooseInLower, alt_score
        else:
            return self.YahtzeeResult.ChooseInUpper, 0

    def choose_joker(self, category, alt_score):
        assert self[category] is None

        if isinstance(category, UpperSection):
            assert alt_score == 0
            self.__upper_section[category] = 0
        else:
            assert alt_score > 0
            self.__lower_section[category] = alt_score

    @property
    def total_score(self):
        return self.upper_section_total + self.lower_section_total

    @property
    def upper_section_total(self):
        upper_total = sum([v for v in self.__upper_section.values() if v is not None])
        upper_total += UPPER_SECTION_BONUS if upper_total >= UPPER_SECTION_BONUS_CONDITION else 0
        return upper_total

    @property
    def lower_section_total(self):
        return sum([v for v in self.__lower_section.values() if v is not None])

    @property
    def is_fulfilled(self):
        have_empty = Scoreboard.__have_empty_category(self.__upper_section) or \
                     Scoreboard.__have_empty_category(self.__lower_section)
        return not have_empty




