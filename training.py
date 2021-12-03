import logging
import os
import pickle
import random
import sys
from typing import List, Any

from chess import WHITE, BLACK, Move

from chessnn import BoardOptim, is_debug, MoveRecord
from chessnn.nn import NNChess
from chessnn.player import NNPLayer, Stockfish


def play_one_game(pwhite, pblack, rnd):
    """

    :type pwhite: NNPLayer
    :type pblack: NNPLayer
    :type rnd: int
    """
    board: BoardOptim = BoardOptim.from_chess960_pos(rnd % 960)
    pwhite.board = board
    pblack.board = board

    try:
        while True:  # and board.fullmove_number < 150
            if not pwhite.makes_move():
                break

            if not pblack.makes_move():
                break

            if is_debug():
                board.write_pgn(pwhite, pblack, os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)
    except:
        last = board.move_stack[-1] if board.move_stack else Move.null()
        logging.warning("Final move: %s %s %s", last, last.from_square, last.to_square)
        logging.warning("Final position:\n%s", board.unicode())
        raise
    finally:
        if board.move_stack:
            board.write_pgn(pwhite, pblack, os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)

    result = board.result(claim_draw=True)

    logging.info("Game #%d/%d:\t%s by %s,\t%d moves", rnd, rnd % 960, result, board.explain(),
                 board.fullmove_number)

    return result


class DataSet(object):
    dataset: List[MoveRecord]

    def __init__(self, fname) -> None:
        super().__init__()
        self.fname = fname
        self.dataset = []

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
                self.dataset.extend(loaded)

        logging.info("Loaded from %s: %s", self.fname, len(self.dataset))

    def update(self, moves):
        lprev = len(self.dataset)
        for move in moves:
            if move.ignore:
                move.forced_eval = 0

        self.dataset.extend(moves)
        if len(self.dataset) - lprev < len(moves):
            logging.debug("partial increase")
        elif len(self.dataset) - lprev == len(moves):
            logging.debug("full increase")
        else:
            logging.debug("no increase")

        while len(self.dataset) > 100000:
            mmin = min(x.from_round for x in self.dataset)
            logging.info("Removing things older than %s", mmin)
            self.dataset = [x for x in self.dataset if x.from_round > mmin]


def set_to_file(draw, param):
    lines = ["%s\n" % item for item in draw]
    lines.sort()
    with open(param, "w") as fhd:
        fhd.writelines(lines)


def play_with_score(pwhite, pblack):
    results = DataSet("results.pkl")
    results.load_moves()

    # if results.dataset:
    #    nn.train(results.dataset, 20)

    rnd = max([x.from_round for x in results.dataset]) if results.dataset else 0
    while True:
        if not ((rnd + 1) % 96):
            results.dump_moves()
            nn.train(results.dataset, 10)
            nn.save("nn.hdf5")

        result = play_one_game(pwhite, pblack, rnd)
        wmoves = pwhite.get_moves(rnd)
        bmoves = pblack.get_moves(rnd)

        if result == '1-0':
            for x, move in enumerate(wmoves):
                move.eval = 0.5 + 0.5 * x / len(wmoves)
                move.from_round = rnd
            for x, move in enumerate(bmoves):
                move.eval = 0.5 - 0.5 * x / len(bmoves)
                move.from_round = rnd

            results.update(wmoves)
            results.update(bmoves)
        elif result == '0-1':
            for x, move in enumerate(wmoves):
                move.eval = 0.5 - 0.5 * x / len(wmoves)
                move.from_round = rnd
            for x, move in enumerate(bmoves):
                move.eval = 0.5 + 0.5 * x / len(bmoves)
                move.from_round = rnd

            results.update(wmoves)
            results.update(bmoves)
        else:
            for x, move in enumerate(wmoves):
                move.eval = 0.5 - 0.25 * x / len(wmoves)
                move.from_round = rnd
            for x, move in enumerate(bmoves):
                move.eval = 0.5 - 0.25 * x / len(bmoves)
                move.from_round = rnd

        nn.train(wmoves + bmoves, 1)

        rnd += 1


def _retrain(winning, losing, draw):
    logging.info("W: %s\tL: %s\tD: %s", len(winning.dataset), len(losing.dataset), len(draw.dataset))
    winning.dump_moves()
    losing.dump_moves()

    lst = list(winning.dataset + losing.dataset)
    random.shuffle(lst)
    if lst:
        nn.train(lst, 20)
        nn.save("nn.hdf5")
        # raise ValueError()

    # winning.dataset.clear()
    # losing.dataset.clear()
    # draw.dataset.clear()


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    _LOG_FORMAT = '[%(relativeCreated)d %(name)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if is_debug() else logging.INFO, format=_LOG_FORMAT)

    # if os.path.exists("nn.hdf5"):
    #    os.remove("nn.hdf5")

    nn = NNChess("nn.hdf5")
    white = NNPLayer("Lisa", WHITE, nn)
    black = NNPLayer("Karen", BLACK, nn)
    # black = Stockfish(BLACK)

    try:
        play_with_score(white, black)
    finally:
        if isinstance(black, Stockfish):
            black.engine.quit()
