import logging
import os
import pickle
import random
import sys
from typing import List

import tensorflow
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
    board: BoardOptim = BoardOptim.from_chess960_pos(random.randint(0, 959))
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
    except BaseException:
        last = board.move_stack[-1] if board.move_stack else Move.null()
        logging.warning("Final move: %s %s %s", last, last.from_square, last.to_square)
        logging.warning("Final position:\n%s", board.unicode())
        raise
    finally:
        if board.move_stack:
            board.write_pgn(pwhite, pblack, os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)

    result = board.result(claim_draw=True)

    avg_invalid = 0
    if isinstance(pwhite, NNPLayer):
        avg_invalid = pwhite.legal_cnt / board.fullmove_number
        pwhite.legal_cnt = 0

    logging.info("Game #%d/%d:\t%s by %s,\t%d moves, legal: %.2f", rnd, rnd % 960, result, board.explain(),
                 board.fullmove_number, avg_invalid)

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
            logging.info("Saving dataset: %s", self.fname)
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

    if results.dataset:
        pass
        # nn.train(results.dataset, 10)
        # nn.save()
        # return

    rnd = 0  # max([x.from_round for x in results.dataset]) if results.dataset else 0
    try:
        while True:
            if not ((rnd + 1) % 96) and len(results.dataset):
                # results.dump_moves()
                nn.train(results.dataset, 1)
                nn.save()
                pass

            if _iteration(pblack, pwhite, results, rnd) != 0:
                # results.dump_moves()
                # nn.save()
                pass

            rnd += 1
    finally:
        results.dump_moves()
        nn.save()


def _iteration(pblack, pwhite, results, rnd) -> int:
    result = play_one_game(pwhite, pblack, rnd)
    wmoves = pwhite.get_moves(rnd)
    bmoves = pblack.get_moves(rnd)

    if result == '1-0':
        for x, move in enumerate(wmoves):
            move.eval = 1  # 0.5 + 0.5 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 0  # 0.5 - 0.5 * x / len(bmoves)
            move.from_round = rnd

        results.update(wmoves)
        nn.train(wmoves, 1)
        # results.update(bmoves)

        return 1
    elif result == '0-1':
        for x, move in enumerate(wmoves):
            move.eval = 0.5 - 0.5 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 1  # 0.5 + 0.5 * x / len(bmoves)
            move.from_round = rnd

        # results.update(wmoves)
        results.update(bmoves)
        nn.train(bmoves, 1)
        return -1
    else:
        for x, move in enumerate(wmoves):
            move.eval = 0.25 + 0.25 * x / len(wmoves)
            move.from_round = rnd
        for x, move in enumerate(bmoves):
            move.eval = 0.25 + 0.25 * x / len(bmoves)
            move.from_round = rnd

        #nn.train(wmoves + bmoves, 1)  # shake it a bit
        return 0


if __name__ == "__main__":
    # sys.setrecursionlimit(10000)
    _LOG_FORMAT = '[%(relativeCreated)d %(name)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if is_debug() else logging.INFO, format=_LOG_FORMAT)
    devices = tensorflow.config.list_physical_devices('GPU')
    logging.info("GPU: %s", devices)
    # assert devices

    nn = NNChess(os.path.join(os.path.dirname(__file__), "models"))
    white = NNPLayer("Lisa", WHITE, nn)
    # white = Stockfish(BLACK)
    black = NNPLayer("Karen", BLACK, nn)
    black = Stockfish(BLACK)

    try:
        play_with_score(white, black)
    finally:
        if isinstance(black, Stockfish):
            black.engine.quit()
