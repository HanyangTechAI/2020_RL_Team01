class Piece:
    def __init__(self, color, name):
        self.name = name
        self.position = None
        self.Color = color

    def isValid(self, startpos, endpos, Color, gameboard):
        if endpos in self.availableMoves(
            startpos[0], startpos[1], gameboard, Color=Color
        ):
            return True
        return False

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def availableMoves(self, x, y, gameboard, Color):
        print("ERROR: no movement for base class")

    def AdNauseum(self, x, y, gameboard, Color, intervals):
        """repeats the given interval until another piece is run into. 
        if that piece is not of the same color, that square is added and
         then the list is returned"""
        answers = []
        for xint, yint in intervals:
            xtemp, ytemp = x + xint, y + yint
            while self.isInBounds(xtemp, ytemp):
                # print(str((xtemp,ytemp))+"is in bounds")

                target = gameboard.get((xtemp, ytemp), None)
                if target is None:
                    answers.append((xtemp, ytemp))
                elif target.Color != Color:
                    answers.append((xtemp, ytemp))
                    break
                else:
                    break

                xtemp, ytemp = xtemp + xint, ytemp + yint
        return answers

    def isInBounds(self, x, y):
        "checks if a position is on the board"
        if x >= 0 and x < 8 and y >= 0 and y < 8:
            return True
        return False

    def noConflict(self, gameboard, initialColor, x, y):
        "checks if a single position poses no conflict to the rules of chess"
        if self.isInBounds(x, y) and (
            ((x, y) not in gameboard) or gameboard[(x, y)].Color != initialColor
        ):
            return True
        return False


chessCardinals = [(1, 0), (0, 1), (-1, 0), (0, -1)]
chessDiagonals = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


def knightList(x, y, int1, int2):
    """sepcifically for the rook, permutes the values needed around a position for noConflict tests"""
    return [
        (x + int1, y + int2),
        (x - int1, y + int2),
        (x + int1, y - int2),
        (x - int1, y - int2),
        (x + int2, y + int1),
        (x - int2, y + int1),
        (x + int2, y - int1),
        (x - int2, y - int1),
    ]


def kingList(x, y):
    return [
        (x + 1, y),
        (x + 1, y + 1),
        (x + 1, y - 1),
        (x, y + 1),
        (x, y - 1),
        (x - 1, y),
        (x - 1, y + 1),
        (x - 1, y - 1),
    ]


class Knight(Piece):
    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        return [
            (xx, yy)
            for xx, yy in knightList(x, y, 2, 1)
            if self.noConflict(gameboard, Color, xx, yy)
        ]


class Rook(Piece):
    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        return self.AdNauseum(x, y, gameboard, Color, chessCardinals)


class Bishop(Piece):
    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        return self.AdNauseum(x, y, gameboard, Color, chessDiagonals)


class Queen(Piece):
    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        return self.AdNauseum(x, y, gameboard, Color, chessCardinals + chessDiagonals)


class King(Piece):
    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        return [
            (xx, yy)
            for xx, yy in kingList(x, y)
            if self.noConflict(gameboard, Color, xx, yy)
        ]


class Pawn(Piece):
    def __init__(self, color, name, direction):
        self.name = name
        self.Color = color
        self.direction = direction

    def availableMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.Color
        answers = []
        if (x + 1, y + self.direction) in gameboard and self.noConflict(
            gameboard, Color, x + 1, y + self.direction
        ):
            answers.append((x + 1, y + self.direction))
        if (x - 1, y + self.direction) in gameboard and self.noConflict(
            gameboard, Color, x - 1, y + self.direction
        ):
            answers.append((x - 1, y + self.direction))
        if (x, y + self.direction) not in gameboard and Color == self.Color:
            answers.append((x, y + self.direction))
        if (
            y in [1, 6]
            and (x, y + self.direction) not in gameboard
            and Color == self.Color
        ):
            answers.append((x, y + 2 * self.direction))

        return answers
