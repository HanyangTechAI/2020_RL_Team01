import numpy as np


class Hai:
    # AI Agent

    def __init__(self):

        # self.color = color
        self.pos_list = [[i, j] for i in range(8) for j in range(8)]

    def input_avail_base(self, avail_list):

        avail_index = np.random.randint(0, len(avail_list))
        loc = avail_list[avail_index]

        return loc

    def input_no_base(self, avail_list):

        while True:

            loc_index = np.random.randint(0, 64)

            loc = self.pos_list[loc_index]

            if loc in avail_list:
                break

        return loc
