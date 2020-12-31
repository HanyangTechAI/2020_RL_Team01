import tkinter as tk
import tkinter.font
from tkinter import messagebox as msg
import numpy as np
from othello import Othello

TILE_SIZE = 32


class gameboard(Othello):
    def __init__(self):
        super().__init__()
        self.white = tk.PhotoImage(file="game_img/white.png")
        self.black = tk.PhotoImage(file="game_img/balck.png")

    def select_mode(self, newGameWindow):

        level = tk.IntVar()
        btn1 = tk.Radiobutton(
            newGameWindow, text="Human vs HAI", variable=level, value=1
        )
        btn2 = tk.Radiobutton(newGameWindow, text="HAI vs HAI", variable=level, value=2)
        btn3 = tk.Radiobutton(
            newGameWindow, text="Human vs DQN_Agent", variable=level, value=3
        )
        btn4 = tk.Radiobutton(
            newGameWindow,
            text="HAI(White) vs DQN_Agent(Black)",
            variable=level,
            value=4,
        )
        btn5 = tk.Radiobutton(
            newGameWindow,
            text="HAI(Black) vs DQN_Agent(White)",
            variable=level,
            value=5,
        )

    def show_board(self):
        pass


def quitGame():
    if msg.askokcancel("오델로", "게임을 종료하시겠습니까?"):
        mainWindow.destroy()


def gameLevelCancel():
    if game is not None:
        game.disabled = False
    newGameWindow.withdraw()


def gameStart(frame):
    winLose.configure(text="")  # 승, 패 안내 초기화
    newGameWindow.withdraw()  # 새 게임 대화상자 닫음
    global game  # 전역변수 game 사용

    return


mainWindow = tk.Tk()
scrW = mainWindow.winfo_screenwidth()
scrH = mainWindow.winfo_screenheight()
mainWindow.geometry(
    "%dx%d+%d+%d"
    % (
        10 * TILE_SIZE,
        10 * TILE_SIZE + 64,
        (scrW - 10 * TILE_SIZE) / 2,
        (scrH - 10 * TILE_SIZE - 64) / 2,
    )
)
mainWindow.resizable(False, False)
mainWindow.title("오델로")
mainWindow.lift()  # mainWindow tk 윈도우를 생성, 초기설정

mainWindow.protocol("WM_DELETE_WINDOW", quitGame)  # 창 닫기 버튼 클릭 시 quitGame 함수 호출

defaultFont = tk.font.Font(family="맑은 고딕", size=10, weight="bold")
mainWindow.option_add("*Font", defaultFont)  # mainWindow 기본 폰트 지정

newGameWindow = tk.Toplevel(mainWindow)
newGameWindow.geometry("344x156+%d+%d" % ((scrW - 344) / 2, (scrH - 156) / 2))
newGameWindow.resizable(False, False)
newGameWindow.title("새 게임")
newGameWindow.wm_attributes("-topmost", 1)  # 새 게임 대화상자 윈도우를 생성, 초기설정
newGameWindow.protocol("WM_DELETE_WINDOW", gameLevelCancel)

levelLabel.grid(column=0, row=0)

level = tk.IntVar()
btn1 = tk.Radiobutton(newGameWindow, text="Human vs HAI", variable=level, value=1)
btn2 = tk.Radiobutton(newGameWindow, text="HAI vs HAI", variable=level, value=2)
btn3 = tk.Radiobutton(newGameWindow, text="Human vs DQN_Agent", variable=level, value=3)
btn4 = tk.Radiobutton(
    newGameWindow, text="HAI(White) vs DQN_Agent(Black)", variable=level, value=4
)
btn5 = tk.Radiobutton(
    newGameWindow, text="HAI(Black) vs DQN_Agent(White)", variable=level, value=5
)

gameStartBtn = tk.Button(
    newGameWindow, text="게임 시작", command=(lambda: gameStart(mainFrame))
)  # 난이도 선택 라디오 버튼 및 게임시작 버튼 생성

btn1.grid(column=0, row=1, padx=16)
btn2.grid(column=0, row=2, padx=16)
btn3.grid(column=0, row=3, padx=16)
btn4.grid(column=0, row=3, padx=16)
btn5.grid(column=0, row=3, padx=16)

gameStartBtn.grid(column=0, row=4, padx=0, pady=16)

# if __name__ == "__main__":

#     othello = gameboard()
#     othello.select_mode()
#     othello.game_run()
