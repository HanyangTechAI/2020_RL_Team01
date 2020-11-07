import unittest
from yahtzee.scoreboard import UpperSection, LowerSection, Scoreboard


class TestScoreboard(unittest.TestCase):
    def test_upper_section(self):
        board = Scoreboard()

        dices = [3, 1, 2, 1, 1]
        board.score(UpperSection.Aces, dices)       # +3
        self.assertEqual(board[UpperSection.Aces], 3)

        dices = [3, 4, 2, 2, 1]
        board.score(UpperSection.Threes, dices)     # +3
        self.assertEqual(board[UpperSection.Threes], 3)

        dices = [3, 4, 2, 2, 6]
        board.score(UpperSection.Fives, dices)      # +0
        self.assertEqual(board[UpperSection.Fives], 0)

        self.assertEqual(board.total_score, 6)

    def test_upper_section_bonus(self):
        board = Scoreboard()

        dices = [3, 1, 1, 1, 2]
        board.score(UpperSection.Aces, dices)       # +3 (3)
        dices = [2, 2, 6, 2, 2]
        board.score(UpperSection.Twos, dices)       # +8 (11)
        dices = [5, 3, 3, 2, 1]
        board.score(UpperSection.Threes, dices)     # +6 (17)
        dices = [4, 1, 4, 4, 2]
        board.score(UpperSection.Fours, dices)      # +12 (29)
        dices = [1, 5, 5, 5, 5]
        board.score(UpperSection.Fives, dices)      # +20 (49)
        dices = [6, 1, 3, 6, 6]
        board.score(UpperSection.Sixes, dices)      # +18 (67)

        self.assertEqual(board.total_score, 67 + 35)

    def test_three_of_a_kind(self):
        board = Scoreboard()

        dices = [3, 1, 3, 3, 2]
        board.score(LowerSection.ThreeOfAKind, dices)   # +12
        self.assertEqual(board[LowerSection.ThreeOfAKind], 12)
        board.reset()

        dices = [1, 1, 1, 1, 4]
        board.score(LowerSection.ThreeOfAKind, dices)   # +8
        self.assertEqual(board[LowerSection.ThreeOfAKind], 8)
        board.reset()

        dices = [1, 2, 3, 2, 4]
        board.score(LowerSection.ThreeOfAKind, dices)   # +0
        self.assertEqual(board[LowerSection.ThreeOfAKind], 0)

    def test_four_of_a_kind(self):
        board = Scoreboard()

        dices = [3, 3, 1, 3, 3]
        board.score(LowerSection.FourOfAKind, dices)    # +13
        self.assertEqual(board[LowerSection.FourOfAKind], 13)
        board.reset()

        dices = [6, 6, 6, 6, 6]
        board.score(LowerSection.FourOfAKind, dices)    # +30
        self.assertEqual(board[LowerSection.FourOfAKind], 30)
        board.reset()

        dices = [1, 1, 2, 2, 1]
        board.score(LowerSection.FourOfAKind, dices)    # +0
        self.assertEqual(board[LowerSection.FourOfAKind], 0)

    def test_full_house(self):
        board = Scoreboard()

        dices = [3, 1, 3, 3, 1]
        board.score(LowerSection.FullHouse, dices)  # +25
        self.assertEqual(board[LowerSection.FullHouse], 25)
        board.reset()

        dices = [3, 3, 1, 1, 2]
        board.score(LowerSection.FullHouse, dices)  # +0
        self.assertEqual(board[LowerSection.FullHouse], 0)

    def test_small_straight(self):
        board = Scoreboard()

        dices = [3, 1, 2, 6, 4]
        board.score(LowerSection.SmallStraight, dices)  # +30
        self.assertEqual(board[LowerSection.SmallStraight], 30)
        board.reset()

        dices = [3, 1, 2, 5, 4]
        board.score(LowerSection.SmallStraight, dices)  # +30
        self.assertEqual(board[LowerSection.SmallStraight], 30)
        board.reset()

        dices = [3, 1, 2, 2, 4]
        board.score(LowerSection.SmallStraight, dices)  # +30
        self.assertEqual(board[LowerSection.SmallStraight], 30)
        board.reset()

        dices = [1, 5, 3, 2, 2]
        board.score(LowerSection.SmallStraight, dices)  # +0
        self.assertEqual(board[LowerSection.SmallStraight], 0)
        board.reset()

        dices = [1, 5, 3, 2, 2]
        board.score(LowerSection.SmallStraight, dices)  # +0
        self.assertEqual(board[LowerSection.SmallStraight], 0)

    def test_large_straight(self):
        board = Scoreboard()

        dices = [4, 5, 3, 1, 2]
        board.score(LowerSection.LargeStraight, dices)  # +40
        self.assertEqual(board[LowerSection.LargeStraight], 40)
        board.reset()

        dices = [4, 5, 3, 1, 3]
        board.score(LowerSection.LargeStraight, dices)  # +0
        self.assertEqual(board[LowerSection.LargeStraight], 0)
        board.reset()

        dices = [2, 5, 3, 1, 1]
        board.score(LowerSection.LargeStraight, dices)  # +0
        self.assertEqual(board[LowerSection.LargeStraight], 0)

    def test_chance(self):
        board = Scoreboard()

        dices = [1, 3, 3, 1, 2]
        board.score(LowerSection.Chance, dices)     # +10
        self.assertEqual(board[LowerSection.Chance], 10)
        board.reset()

        dices = [1, 1, 1, 1, 6]
        board.score(LowerSection.Chance, dices)     # +10
        self.assertEqual(board[LowerSection.Chance], 10)

    def test_yahtzee(self):
        board = Scoreboard()

        dices = [1, 1, 1, 1, 1]
        result, alt_score = board.score_yahtzee(dices)   # +50
        self.assertEqual(board[LowerSection.Yahtzee], 50)
        self.assertEqual(result, Scoreboard.YahtzeeResult.Scored)
        self.assertEqual(alt_score, None)
        board.reset()

        dices = [1, 1, 1, 2, 1]
        result, alt_score = board.score_yahtzee(dices)  # +0
        self.assertEqual(board[LowerSection.Yahtzee], 0)
        self.assertEqual(result, Scoreboard.YahtzeeResult.Scored)
        self.assertEqual(alt_score, None)
        board.reset()

        dices = [2, 2, 2, 2, 2]
        result, alt_score = board.score_yahtzee(dices)   # +50
        self.assertEqual(board[LowerSection.Yahtzee], 50)
        self.assertEqual(result, Scoreboard.YahtzeeResult.Scored)
        self.assertEqual(alt_score, None)
        dices = [2, 2, 2, 2, 2]
        result, alt_score = board.score_yahtzee(dices)   # bonus +100, twos +10
        self.assertEqual(board[LowerSection.Yahtzee], 50)
        self.assertEqual(board[LowerSection.YahtzeeBonus], 100)
        self.assertEqual(board[UpperSection.Twos], 10)
        self.assertEqual(result, Scoreboard.YahtzeeResult.Scored)
        self.assertEqual(alt_score, None)
        dices = [2, 2, 2, 2, 2]
        result, alt_score = board.score_yahtzee(dices)   # bonus +100
        self.assertEqual(board[LowerSection.Yahtzee], 50)
        self.assertEqual(board[LowerSection.YahtzeeBonus], 200)
        self.assertEqual(board[UpperSection.Twos], 10)
        self.assertEqual(result, Scoreboard.YahtzeeResult.ChooseInLower)
        self.assertEqual(alt_score, 10)
        board.choose_joker(LowerSection.FullHouse, alt_score)     # fullhouse +10
        self.assertEqual(board[LowerSection.FullHouse], 10)
        board.reset()

        dices = [1, 1, 1, 1, 2]
        board.score_yahtzee(dices)    # +0
        self.assertEqual(board[LowerSection.Yahtzee], 0)
        # fill lower section except Chance
        board.score(LowerSection.ThreeOfAKind, dices)   # +6
        board.score(LowerSection.FourOfAKind, dices)    # +6
        board.score(LowerSection.FullHouse, dices)      # +0
        board.score(LowerSection.SmallStraight, dices)  # +0
        board.score(LowerSection.LargeStraight, dices)  # +0
        dices = [1, 1, 1, 1, 1]
        result, alt_score = board.score_yahtzee(dices)    # bonus +0
        self.assertEqual(board[LowerSection.YahtzeeBonus], 0)
        self.assertEqual(board[UpperSection.Aces], 5)
        self.assertEqual(result, Scoreboard.YahtzeeResult.Scored)
        self.assertEqual(alt_score, None)
        result, alt_score = board.score_yahtzee(dices)   # bonus +0
        self.assertEqual(board[LowerSection.YahtzeeBonus], 0)
        self.assertEqual(result, Scoreboard.YahtzeeResult.ChooseInLower)
        self.assertEqual(alt_score, 5)
        board.choose_joker(LowerSection.Chance, alt_score)
        self.assertEqual(board[LowerSection.Chance], 5)
        result, alt_score = board.score_yahtzee(dices)   # bonus +0
        self.assertEqual(board[LowerSection.YahtzeeBonus], 0)
        self.assertEqual(result, Scoreboard.YahtzeeResult.ChooseInUpper)
        self.assertEqual(alt_score, 0)
        board.choose_joker(UpperSection.Twos, alt_score)
        self.assertEqual(board[UpperSection.Twos], 0)








