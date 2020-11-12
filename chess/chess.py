from piece import Pawn, Rook, Knight, Bishop, King, Queen

WHITE = "white"
BLACK = "black"

uniDict = {
    WHITE: {Pawn: "♙", Rook: "♖", Knight: "♘", Bishop: "♗", King: "♔", Queen: "♕"},
    BLACK: {Pawn: "♟", Rook: "♜", Knight: "♞", Bishop: "♝", King: "♚", Queen: "♛"},
}


class Game:
    def __init__(self):
        self.playersturn = BLACK
        self.message = "this is where prompts will go"
        self.gameboard = {}

    def placePieces(self):

        for i in range(0, 8):
            self.gameboard[(i, 1)] = Pawn(WHITE, uniDict[WHITE][Pawn], 1)
            self.gameboard[(i, 6)] = Pawn(BLACK, uniDict[BLACK][Pawn], -1)

        white_placers = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        black_placers = [Rook, Knight, Bishop, King, Queen, Bishop, Knight, Rook]

        for i in range(0, 8):
            self.gameboard[(i, 0)] = white_placers[i](
                WHITE, uniDict[WHITE][white_placers[i]]
            )
            self.gameboard[((7 - i), 7)] = black_placers[i](
                BLACK, uniDict[BLACK][black_placers[i]]
            )

    def main(self):

        while True:
            self.printBoard()
            print(self.message)
            if self.playersturn == "white":
                print("white's turn")
            else:
                print("black's turn")
            self.message = ""
            startpos, endpos = self.parseInput()
            # exit method
            if startpos == "exit" and endpos == "game":
                if input("Are you want to exit game(reply yes or no)") == "yes":
                    break

            try:
                target = self.gameboard[startpos]
            except:
                self.message = "could not find piece; index probably out of range"
                target = None


            if target:
                print("found " + str(target))
                if target.Color != self.playersturn:
                    self.message = "you aren't allowed to move that piece this turn"
                    continue
                if target.isValid(startpos, endpos, target.Color, self.gameboard):
                    self.message = "that is a valid move"
                    self.gameboard[endpos] = self.gameboard[startpos]
                    del self.gameboard[startpos]
                    if self.gameover():
                        print(f"{self.playersturn} win the game")
                        break

                    self.isCheck()
                    if self.playersturn == BLACK:
                        self.playersturn = WHITE
                    else:
                        self.playersturn = BLACK
                else:
                    self.message = "invalid move" + str(
                        target.availableMoves(startpos[0], startpos[1], self.gameboard)
                    )
                    # print(
                    #     target.availableMoves(startpos[0], startpos[1], self.gameboard)
                    # )
            else:
                self.message = "there is no piece in that space"

    def gameover(self):
        count=0
        for _, piece in self.gameboard.items():
            if type(piece) == King:
                count +=1

        return count != 2

    def isCheck(self):
        kingDict = {}
        pieceDict = {BLACK: [], WHITE: []}
        for position, piece in self.gameboard.items():
            if type(piece) == King:
                kingDict[piece.Color] = position
            # print(piece)
            pieceDict[piece.Color].append((piece, position))
        # white
        if self.canSeeKing(kingDict[WHITE], pieceDict[BLACK]):
            self.message = "White player is in check"
        if self.canSeeKing(kingDict[BLACK], pieceDict[WHITE]):
            self.message = "Black player is in check"

    def canSeeKing(self, kingpos, piecelist):
        # checks if any pieces in piece list (which is an array of (piece,position) tuples) can see the king in kingpos
        for piece, position in piecelist:
            if piece.isValid(position, kingpos, piece.Color, self.gameboard):
                return True

    def parseInput(self):
        try:
            a, b = input().split()
            # exit method
            if a == "exit" and b == "game":
                return (a, b)

            a = ((ord(a[0]) - 97), int(a[1]) - 1)
            b = (ord(b[0]) - 97, int(b[1]) - 1)
            return (a, b)
        except:
            print("error decoding input. please try again")
            return ((-1, -1), (-1, -1))

    """def validateInput(self, *kargs):
        for arg in kargs:
            if type(arg[0]) is not type(1) or type(arg[1]) is not type(1):
                return False
        return True"""

    def printBoard(self):
        print("   1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")
        for i in range(0, 8):
            print("-" * 40)
            print(chr(i + 97), end="|")
            for j in range(0, 8):
                item = self.gameboard.get((i, j), " ")
                print(str(item) + " |", end=" ")
            print(chr(i + 97))
        print("-" * 40)
        print("   1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")

    def run(self):
        self.placePieces()
        print("chess start. if you want to exit game type 'exit game'")
        print("left player is white right player is black")
        self.main()


if __name__ == "__main__":

    game = Game()
    game.run()
