import json
import logging
import os
import pickle
import sys
from collections import Counter

from chess import STARTING_FEN, Board, pgn, WHITE, BLACK

from nn import NN
from player import Player


class MyStringExporter(pgn.StringExporter):

    def __init__(self, comments):
        super().__init__(headers=True, variations=True, comments=True)
        self.comm_stack = comments

    def visit_move(self, board, move):
        if self.variations or not self.variation_depth:
            # Write the move number.
            if board.turn == WHITE:
                self.write_token(str(board.fullmove_number) + ". ")
            elif self.force_movenumber:
                self.write_token(str(board.fullmove_number) + "... ")

            # Write the SAN.
            self.write_token(board.san(move) + " {%s} " % self.comm_stack.pop(0))

            self.force_movenumber = False


def record_results(brd, rnd, avgs):
    journal = pgn.Game.from_board(brd)
    journal.headers.clear()
    journal.headers["White"] = "Lisa"
    journal.headers["Black"] = "Karen"
    journal.headers["Round"] = rnd
    journal.headers["Result"] = brd.result(claim_draw=True)
    if brd.is_checkmate():
        comm = "checkmate"
    elif brd.can_claim_fifty_moves():
        comm = "50 moves"
    elif brd.can_claim_threefold_repetition():
        comm = "threefold"
    elif brd.is_insufficient_material():
        comm = "material"
    elif not any(brd.generate_legal_moves()):
        comm = "stalemate"
    else:
        comm = "by other reason"
    journal.headers["Site"] = comm

    exporter = MyStringExporter(brd.comment_stack)
    pgns = journal.accept(exporter)
    # logging.info("\n%s", pgns)
    logging.info("Game #%d:\t%s by %s,\t%d moves,\t%.3f AMS", rnd, journal.headers["Result"], comm,
                 brd.fullmove_number, avgs)
    with open("last.pgn", "w") as out:
        out.write(pgns)


class BoardOptim(Board):

    def __init__(self, fen=STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self._fens = []
        self.comment_stack = []

    def can_claim_threefold_repetition1(self):
        # repetition = super().can_claim_threefold_repetition()
        # if repetition:
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 3

    def is_fivefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5

    def push1(self, move):
        super().push(move)
        self._fens.append(self.epd().replace(" w ", " . ").replace(" b ", " . "))

    def pop1(self):
        self._fens.pop(-1)
        return super().pop()


def play_one_game(pwhite, pblack, rnd):
    board = BoardOptim(STARTING_FEN)
    pwhite.board = board
    pblack.board = board

    while pwhite.makes_move() and pblack.makes_move():  # and board.fullmove_number < 150
        # logging.debug("%s. %s %s", board.fullmove_number - 1, board.move_stack[-1], board.move_stack[-2])
        pass

    all_moves = pwhite.moves_log + pblack.moves_log
    avg_score = sum([x.get_score() for x in all_moves]) / float(len(all_moves))
    record_results(board, rnd, avg_score)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG)

    nn = NN("nn.hdf5")
    white = Player(WHITE, nn)
    black = Player(BLACK, nn)

    dataset = set()
    if os.path.exists("moves.pkl"):
        with open("moves.pkl", 'rb') as fhd:
            dataset.update(pickle.load(fhd))

    rnd = 0
    while True:
        rnd += 1
        play_one_game(white, black, rnd)

        wmoves = white.get_moves()
        bmoves = black.get_moves()
        game_data = wmoves + bmoves
        dataset.update(game_data)

        with open("moves.pkl", "wb") as fhd:
            pickle.dump(list(dataset), fhd)

        if not (rnd % 20):
            nn.learn(dataset, 10)
            nn.save("nn.hdf5")
