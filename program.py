import logging
import sys
from collections import Counter

from chess import STARTING_FEN, Board, pgn, WHITE, BLACK

from nn import NN
from player import Player


def record_results(brd, rnd):
    journal = pgn.Game.from_board(brd)
    journal.headers.clear()
    journal.headers["White"] = "Lisa"
    journal.headers["Black"] = "Karen"
    journal.headers["Round"] = rnd
    journal.headers["Result"] = brd.result(claim_draw=True)
    if brd.is_checkmate():
        journal.end().comment = "checkmate"
    elif brd.can_claim_fifty_moves():
        journal.end().comment = "50 moves claim"
    elif brd.can_claim_threefold_repetition():
        journal.end().comment = "threefold claim"
    elif brd.is_insufficient_material():
        journal.end().comment = "insufficient material"
    elif not any(brd.generate_legal_moves()):
        journal.end().comment = "stalemate"
    else:
        journal.end().comment = "by other reason"

    # exporter = pgn.StringExporter(headers=True, variations=True, comments=True)
    # logging.info("\n%s", journal.accept(exporter))
    logging.info("Game #%d: %s by %s, %d moves", rnd, journal.headers["Result"], journal.end().comment,
                 brd.fullmove_number)
    with open("last.pgn", "w") as out:
        exporter = pgn.FileExporter(out)
        journal.accept(exporter)


class BoardOptim(Board):

    def __init__(self, fen=STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self._fens = []

    def can_claim_threefold_repetition(self):
        # repetition = super().can_claim_threefold_repetition()
        # if repetition:
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 3

    def is_fivefold_repetition(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5

    def push(self, move):
        super().push(move)
        self._fens.append(self.epd().replace(" w ", " . ").replace(" b ", " . "))

    def pop(self):
        self._fens.pop(-1)
        return super().pop()


def play_one_game(pwhite, pblack, rnd):
    board = BoardOptim(STARTING_FEN)
    pwhite.board = board
    pblack.board = board

    while pwhite.makes_move() and pblack.makes_move():  # and board.fullmove_number < 150
        # logging.debug("%s. %s %s", board.fullmove_number - 1, board.move_stack[-1], board.move_stack[-2])
        pass

    record_results(board, rnd)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG)

    nn = NN("nn.hdf5")
    white = Player(WHITE, nn)
    black = Player(BLACK, nn)

    rnd = 0
    useful_stack = []
    while True:
        rnd += 1
        play_one_game(white, black, rnd)

        game_data = white.get_moves() + black.get_moves()
        if game_data[0]["result"] != 0.5:
            useful_stack.append(game_data)

        if not (rnd % 10):
            data = []
            if not useful_stack:
                data.extend(game_data)
            for item in useful_stack:
                data.extend(item)

            nn.learn(data, 10 if game_data[0]["result"] == 0.5 else 10)
            nn.save("nn.hdf5")
