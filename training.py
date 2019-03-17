import logging
import os
import pickle
import sys

from chess import STARTING_FEN, WHITE, BLACK

from chessnn import BoardOptim
from chessnn.nn import NN
from chessnn.player import Player


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

            # break
            nn.learn(dataset, 20)
            # nn.save("nn.hdf5")
