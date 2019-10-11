import logging
import os
import pickle
import random
import sys

from chess import WHITE, BLACK, Move

from chessnn import BoardOptim, is_debug
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
            if not pwhite.makes_move(rnd):
                break
            if not pblack.makes_move(rnd):
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

    badp = 0
    badc = 0
    if isinstance(pwhite, NNPLayer):
        badp += pwhite.invalid_moves
        pwhite.invalid_moves = 0
        badc += 1

    if isinstance(pblack, NNPLayer):
        badp += pblack.invalid_moves
        pblack.invalid_moves = 0
        badc += 1

    badp = badp / badc

    logging.info("Game #%d/%d:\t%s by %s,\t%d moves, %d%% bad", rnd, rnd % 960, result, board.explain(),
                 board.fullmove_number, badp)

    return result


class DataSet(object):
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

        while len(self.dataset) > 50000:
            mmin = min([x.from_round for x in self.dataset])
            logging.info("Removing things older than %s", mmin)
            for x in list(self.dataset):
                if x.from_round <= mmin:
                    self.dataset.remove(x)


def set_to_file(draw, param):
    lines = ["%s\n" % item for item in draw]
    lines.sort()
    with open(param, "w") as fhd:
        fhd.writelines(lines)


def play_with_score(pwhite, pblack):
    winning = DataSet("winning.pkl")
    winning.load_moves()
    losing = DataSet("losing.pkl")
    losing.load_moves()
    draw = DataSet("losing.pkl")

    rnd = max([x.from_round for x in winning.dataset + losing.dataset]) if winning.dataset else 0
    while True:
        if not ((rnd+1) % 960):
            _retrain(winning, losing, draw)

        result = play_one_game(pwhite, pblack, rnd)
        wmoves = pwhite.get_moves()
        bmoves = pblack.get_moves()
        good_moves = _fill_sets(result, wmoves, bmoves, losing, winning, draw)
        if good_moves and True:
            moves = wmoves + bmoves
            random.shuffle(moves)
            nn.train(moves, 1)
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
        #raise ValueError()

    # winning.dataset.clear()
    # losing.dataset.clear()
    # draw.dataset.clear()


def _fill_sets(result, wmoves, bmoves, losing, winning, draw):
    if result == '1-0':
        # playsound.playsound('/usr/share/games/xboard/sounds/ding.wav')
        for x, move in enumerate(wmoves):
            move.eval = 1.0
        for x, move in enumerate(bmoves):
            move.eval = 0.0
        winning.update(wmoves)
        losing.update(bmoves)
        return wmoves
    elif result == '0-1':
        for x, move in enumerate(bmoves):
            move.eval = 1.0
        for x, move in enumerate(wmoves):
            move.eval = 0.0
        winning.update(bmoves)
        losing.update(wmoves)
        return bmoves
    else:
        for x, move in enumerate(bmoves):
            move.eval = 0.5
        for x, move in enumerate(wmoves):
            move.eval = 0.5
        # draw.update(wmoves)
        # draw.update(bmoves)
        return []


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG if is_debug() else logging.INFO)

    # if os.path.exists("nn.hdf5"):
    #    os.remove("nn.hdf5")

    nn = NNChess("nn.hdf5")
    white = NNPLayer("Lisa", WHITE, nn)
    black = NNPLayer("Karen", BLACK, nn)
    black = Stockfish(BLACK)

    try:
        play_with_score(white, black)
    finally:
        if isinstance(black, Stockfish):
            black.engine.quit()
