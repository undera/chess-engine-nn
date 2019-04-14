import logging
import os
import pickle
import sys
from typing import Set

from chess import STARTING_FEN, WHITE, BLACK

from chessnn import BoardOptim, MoveRecord
from chessnn.nn import NN
from chessnn.player import Player


def play_one_game(pwhite, pblack, rnd):
    """

    :type pwhite: Player
    :type pblack: Player
    :type rnd: int
    """
    board = BoardOptim(STARTING_FEN)
    pwhite.board = board
    pwhite.start_from = rnd % 20
    pblack.board = board

    while True:  # and board.fullmove_number < 150
        if not pwhite.makes_move():
            break
        if not pblack.makes_move():
            break

    board.write_pgn(os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)

    avg_score_w = sum([x.get_score() for x in pwhite.moves_log]) / float(len(pwhite.moves_log))
    avg_score_b = sum([x.get_score() for x in pblack.moves_log]) / float(len(pblack.moves_log))
    logging.info("Game #%d:\t%s by %s,\t%d moves,\t%.2f / %.2f AMS", rnd, board.result(claim_draw=True),
                 board.explain(), board.fullmove_number, avg_score_w, avg_score_b)

    return board.result(claim_draw=True)


class DataSet(object):
    def __init__(self, fname) -> None:
        super().__init__()
        self.fname = fname
        self.dataset = set()

    def dump_moves(self):
        if os.path.exists(self.fname):
            os.rename(self.fname, self.fname + ".bak")
        try:
            with open(self.fname, "wb") as fhd:
                pickle.dump(self.dataset, fhd)
        except:
            os.rename(self.fname + ".bak", self.fname)

    def load_moves(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'rb') as fhd:
                loaded = pickle.load(fhd)
                self.dataset.update(loaded)

    def update(self, moves):
        lprev = len(self.dataset)
        self.dataset.update(moves)
        if len(self.dataset) - lprev < len(moves):
            logging.debug("partial increase")
        elif len(self.dataset) - lprev == len(moves):
            logging.debug("full increase")
        else:
            logging.debug("no increase")


def set_to_file(draw, param):
    lines = ["%s\n" % item for item in draw]
    lines.sort()
    with open(param, "w") as fhd:
        fhd.writelines(lines)


def play_with_score(pwhite, pblack):
    winning: Set[MoveRecord] = set()
    losing: Set[MoveRecord] = set()
    draw: Set[MoveRecord] = set()

    rnd = 0
    while True:
        result = play_one_game(pwhite, pblack, rnd)

        wmoves = pwhite.get_moves()
        bmoves = pblack.get_moves()

        if result == '1-0':
            winning.update(wmoves)
            losing.update(bmoves)
        elif result == '0-1':
            winning.update(bmoves)
            losing.update(wmoves)
        else:
            draw.update(wmoves)
            draw.update(bmoves)

        rnd += 1
        if not (rnd % 20):
            winning -= losing
            winning -= draw
            losing -= winning
            losing -= draw
            logging.info("Orig: %s %s %s", len(winning), len(losing), len(draw))

            if not winning and not losing:
                nn.learn(draw, 1)
            else:
                for x in winning:
                    x.forced_score = 1.0
                for x in losing:
                    x.forced_score = 0.0

                # dataset.update(pure_win)
                # dataset.update(pure_loss)
                # dataset.dump_moves()
                nn.learn(winning | losing, 20)
                # nn.save("nn.hdf5")


def play_per_turn(pwhite, pblack):
    dataset = DataSet("moves.pkl")
    dataset.load_moves()
    pwhite.nn.learn(dataset.dataset, 50)
    nn.save("nn.hdf5")

    rnd = 0
    while True:
        result = play_one_game(pwhite, pblack, rnd)

        dataset.update(pwhite.get_moves() + pblack.get_moves())

        rnd += 1
        if not (rnd % 20):
            # dataset.dump_moves()

            # break
            nn.learn(dataset.dataset, 50)
            # nn.save("nn.hdf5")
            break


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.INFO)

    nn = NN("nn.hdf5")
    white = Player(WHITE, nn)
    black = Player(BLACK, nn)

    play_per_turn(white, black)
