import logging
import os
import pickle
import random
import sys
from typing import Set

from chess import WHITE, BLACK

from chessnn import BoardOptim, MoveRecord, is_debug
from chessnn.nn import NNChess
from chessnn.player import Player

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def play_one_game(pwhite, pblack, rnd):
    """

    :type pwhite: Player
    :type pblack: Player
    :type rnd: int
    """
    board = BoardOptim.from_chess960_pos(random.randint(0, 959))
    pwhite.board = board
    pblack.board = board

    while True:  # and board.fullmove_number < 150
        if not pwhite.makes_move(rnd):
            break
        if not pblack.makes_move(rnd):
            break

    board.write_pgn(os.path.join(os.path.dirname(__file__), "last.pgn"), rnd)

    avg_score_w = sum([x.get_eval() for x in pwhite.moves_log]) / float(len(pwhite.moves_log))
    avg_score_b = sum([x.get_eval() for x in pblack.moves_log]) / float(len(pblack.moves_log))
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
        for move in moves:
            if move.ignore:
                move.forced_eval = 0

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
    winning = DataSet("winning.pkl")
    winning.load_moves()
    losing = DataSet("losing.pkl")
    losing.load_moves()
    draw: Set[MoveRecord] = set()

    # if not is_debug():
    #    nn.learn(winning.dataset, 20)

    rnd = max([x.from_round for x in winning.dataset | losing.dataset]) if winning.dataset else 0
    non_decisive_cnt = 0
    had_decisive = False
    while True:
        result = play_one_game(pwhite, pblack, rnd)

        wmoves = pwhite.get_moves()
        bmoves = pblack.get_moves()

        if result == '1-0':
            had_decisive = True
            for x, move in enumerate(wmoves):
                move.forced_eval = 0.5 + 0.5 * float(x) / len(wmoves)
            for x, move in enumerate(bmoves):
                move.forced_eval = 0.5 - 0.5 * float(x) / len(wmoves)
            winning.update(wmoves)
            losing.update(bmoves)
        elif result == '0-1':
            for x, move in enumerate(bmoves):
                move.forced_eval = 0.5 + 0.5 * float(x) / len(wmoves)
            for x, move in enumerate(wmoves):
                move.forced_eval = 0.5 - 0.5 * float(x) / len(wmoves)
            had_decisive = True
            winning.update(bmoves)
            losing.update(wmoves)
        else:
            draw.update(wmoves)
            draw.update(bmoves)

        rnd += 1
        if not (rnd % 20) or False:
            # if had_decisive:
            # winning.dataset -= losing.dataset
            # winning.dataset -= draw
            # losing.dataset -= winning.dataset
            # losing.dataset -= draw

            if not had_decisive:
                non_decisive_cnt += 1
            else:
                non_decisive_cnt = 0

            while len(winning.dataset) > 10000:
                mmin = min([x.from_round for x in winning.dataset])
                logging.info("Removing from winning things older than %s", mmin)
                for x in list(winning.dataset):
                    if x.from_round <= mmin:
                        winning.dataset.remove(x)

            while len(losing.dataset) > 10000:
                mmin = min([x.from_round for x in losing.dataset])
                logging.info("Removing from losing things older than %s", mmin)
                for x in list(losing.dataset):
                    if x.from_round <= mmin:
                        losing.dataset.remove(x)

            logging.info("W: %s\tL: %s\tD: %s\tNon-dec: %s", len(winning.dataset), len(losing.dataset), len(draw),
                         non_decisive_cnt)

            winning.dump_moves()
            losing.dump_moves()
            dataset = winning.dataset | losing.dataset

            lst = list(draw)
            for x in lst:
                x.forced_eval = 0.5 if not x.ignore else 0  # random.random()
            random.shuffle(lst)
            # dataset.update(lst[:10 * non_decisive_cnt])
            dataset.update(lst[:max(10 * non_decisive_cnt + 1, len(dataset) // 2000)])

            if had_decisive or not non_decisive_cnt % 5:
                nn.train(dataset, 20)
                nn.save("nn.hdf5")

            draw = set()
            had_decisive = False


def play_per_turn(pwhite, pblack):
    dataset = DataSet("moves.pkl")
    dataset.load_moves()
    if not is_debug():
        pwhite.nn.learn(dataset.dataset, 20)
        nn.save("nn.hdf5")

    rnd = max([x.from_round for x in dataset.dataset]) if dataset.dataset else 0
    while True:
        result = play_one_game(pwhite, pblack, rnd)

        moves = pwhite.get_moves() + pblack.get_moves()
        moves = list(filter(lambda x: x.get_eval() > 0, moves))
        dataset.update(moves)

        rnd += 1
        if not (rnd % 20):
            dataset.dump_moves()

            nn.train(dataset.dataset, 20)
            nn.save("nn.hdf5")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG if is_debug() else logging.INFO)

    nn = NNChess()
    white = Player(WHITE, nn)
    black = Player(BLACK, nn)

    # play_per_turn(white, black)
    play_with_score(white, black)
