# Inspired by TCEC Season 14 - Superfinal
# where Leela was trying to fry Stockfish
import math
import numpy

import chess

from nn import NN


class Player(object):
    board: chess.Board
    nn: NN

    def __init__(self, brd, piece_index) -> None:
        super().__init__()
        self.piece_index = piece_index
        self.board = brd
        self.nn = NN()

    def get_move(self):
        weights_from, weights_to = self.nn.query(self.board)
        move_rating = []
        for move in self.board.generate_legal_moves():
            sr = move.from_square // 8
            sf = move.from_square % 8
            sw = weights_from[sr][sf]

            dr = move.to_square // 8
            df = move.to_square % 8
            dw = weights_to[dr][df]

            score = abs(sw) * abs(dw) * numpy.sign(min(sw, dw))

            move_rating.append((move, score))

        move_rating.sort(key=lambda x: x[1], reverse=True)
        while move_rating:
            move = move_rating.pop(0)[0]
            return move
