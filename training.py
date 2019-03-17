import copy
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
        self.comm_stack = copy.copy(comments)

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


class BoardOptim(Board):

    def __init__(self, fen=STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self._fens = []
        self.comment_stack = []

    def write_pgn(self, fname, roundd):
        journal = pgn.Game.from_board(self)
        journal.headers.clear()
        journal.headers["White"] = "Lisa"
        journal.headers["Black"] = "Karen"
        journal.headers["Round"] = roundd
        journal.headers["Result"] = self.result(claim_draw=True)
        journal.headers["Site"] = self.explain()
        exporter = MyStringExporter(self.comment_stack)
        pgns = journal.accept(exporter)
        with open(fname, "w") as out:
            out.write(pgns)

    def explain(self):
        if self.is_checkmate():
            comm = "checkmate"
        elif self.can_claim_fifty_moves():
            comm = "50 moves"
        elif self.can_claim_threefold_repetition():
            comm = "threefold"
        elif self.is_insufficient_material():
            comm = "material"
        elif not any(self.generate_legal_moves()):
            comm = "stalemate"
        else:
            comm = "by other reason"
        return comm

    def can_claim_threefold_repetition1(self):
        # repetition = super().can_claim_threefold_repetition()
        # if repetition:
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 3

    def is_fivefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5

    def can_claim_draw1(self):
        return super().can_claim_draw() or self.fullmove_number > 100

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

    while True:  # and board.fullmove_number < 150
        if not pwhite.makes_move():
            break
        if not pblack.makes_move():
            break

    board.write_pgn(os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)

    all_moves = pwhite.moves_log + pblack.moves_log
    avg_score = sum([x.get_score() for x in all_moves]) / float(len(all_moves))
    logging.info("Game #%d:\t%s by %s,\t%d moves,\t%.3f AMS", rnd, board.result(claim_draw=True), board.explain(),
                 board.fullmove_number, avg_score)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG)

    nn = NN("nn.hdf5")
    white = Player(WHITE, nn)
    black = Player(BLACK, nn)

    dataset = set()
    if os.path.exists("moves.pkl"):
        with open("moves.pkl", 'rb') as fhd:
            loaded = pickle.load(fhd)
            dataset.update(loaded)
            nn.learn(dataset, 50)
            # nn.save("nn.hdf5")

    rnd = 0
    while True:
        rnd += 1
        play_one_game(white, black, rnd)

        wmoves = white.get_moves()
        bmoves = black.get_moves()
        game_data = wmoves + bmoves
        l1 = len(dataset)
        dataset.update(game_data)
        # logging.info("%s+%s=%s", l1, len(game_data), len(dataset))
        assert len(dataset) < l1 + len(game_data)

        if not (rnd % 20):
            if os.path.exists("moves.pkl"):
                os.rename("moves.pkl", "moves.bak.pkl")
            try:
                with open("moves.pkl", "wb") as fhd:
                    pickle.dump(dataset, fhd)
            except:
                os.rename("moves.bak.pkl", "moves.pkl")

            #break
            nn.learn(dataset, 20)
            # nn.save("nn.hdf5")
