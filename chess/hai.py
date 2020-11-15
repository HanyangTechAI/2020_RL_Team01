from random import choice

from piece import Bishop, King, Knight, Pawn, Queen, Rook

class Hai:
    def __init__(self):
        self.playersturn = None
        self.gameboard = None
        self.Color = None

    def autoInput(self, Color):
        piecelist = []
        for position, piece in self.gameboard.items():
            if piece.Color == self.playersturn:
                piecelist.append((piece, position))
        for _ in range(100):
            rand_start = choice(piecelist)
            position = rand_start[1]
            piece = rand_start[0]
            available_list = piece.availableMoves(
                position[0], position[1], self.gameboard, Color
            )
            if not available_list:
                continue
            rand_end = choice(available_list)

            print("computer played", position, rand_end)

            return position, rand_end