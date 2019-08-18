import logging
import random

import chess
from chess import WHITE
from chess.engine import SimpleEngine

from chessnn import BoardOptim, is_debug
from chessnn.nn import NNChess
from chessnn.player import NNPLayer

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if is_debug() else logging.INFO)

    engine = SimpleEngine.popen_uci("stockfish")

    try:
        board = BoardOptim.from_chess960_pos(random.randint(0, 959))
        nn = NNChess("nn.hdf5")
        white = NNPLayer("Lisa", WHITE, nn)
        white.board = board

        while not board.is_game_over():
            if not white.makes_move(0):
                break

            result = engine.play(board, chess.engine.Limit(time=0.100))
            board.push(result.move)

        logging.info("Result: %s", board.result())
    finally:
        engine.quit()
