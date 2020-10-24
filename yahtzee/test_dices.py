import unittest
from .dices import Dices


class TestDices(unittest.TestCase):
    def test_rolling(self):
        dices = Dices()

        self.assertEqual(dices.have_rolled, False)
        self.assertEqual(dices.can_roll, True)

        dices.roll_all()
        self.assertEqual(dices.have_rolled, True)
        self.assertEqual(dices.can_roll, True)

        select = [1, 3]
        v_0, v_2, v_4 = dices[0], dices[2], dices[4]
        dices.roll(select)
        self.assertEqual(v_0, dices[0])
        self.assertEqual(v_2, dices[2])
        self.assertEqual(v_4, dices[4])
        self.assertEqual(dices.have_rolled, True)
        self.assertEqual(dices.can_roll, True)

        select = [0, 2, 3, 4]
        v_1 = dices[1]
        dices.roll(select)
        self.assertEqual(v_1, dices[1])
        self.assertEqual(dices.have_rolled, True)
        self.assertEqual(dices.can_roll, False)

        dices.reset()
        self.assertEqual(dices.have_rolled, False)
        self.assertEqual(dices.can_roll, True)

    def test_get_array(self):
        dices = Dices()

        dices.roll_all()
        arr = dices.array
        for i in range(len(dices)):
            self.assertEqual(arr[i], dices[i])

        dices.roll_all()
        arr = dices.array
        for i in range(len(dices)):
            self.assertEqual(arr[i], dices[i])

