# Inspired by TCEC Season 14 - Superfinal
# where Leela was trying to fry Stockfish

import logging

import keras
import numpy as np
from keras import Input
from keras.layers import Dense
from keras.utils import plot_model

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECE_MAP = "PpNnBbRrQqKk"


class Board(object):
    def __init__(self) -> None:
        super().__init__()
        self.piece_placement = np.full((8, 8, 12), 0)  # rank, col, piece kind
        self._50move_counter = 0
        self.move_num = 0
        self.active_index = 0
        self.castling_flags = "KQkq"
        self.moves = []
        self.starting_fen = None

    def is_playable(self):
        return True

    def make_move(self, move):
        # check who's move it is
        # tick active_index and move_num
        logging.info("New Move")
        self.moves.append(move)
        # check 3-fold and 50-move

    def from_fen(self, fen):
        self.starting_fen = fen
        placement, active_colour, self.castling_flags, enpassant, halfmove, fullmove = fen.split(' ')
        self._50move_counter = int(halfmove)
        self.move_num = fullmove
        self.active_index = 0 if active_colour == 'w' else 1

        rankn = 8
        for rank in placement.split('/'):
            rankn -= 1
            coln = 0
            for col in rank:
                try:
                    coln += int(col)
                except:
                    cell = self.piece_placement[rankn][coln]
                    cell[PIECE_MAP.index(col)] = 1
                    coln += 1

            assert coln == 8
        assert rankn == 0


class NN(object):
    def __init__(self) -> None:
        super().__init__()
        self._model = self._get_nn()
        self._model.summary(print_fn=logging.warning)

    def _get_nn(self):
        positions = Input(shape=(8 * 8 * len(PIECE_MAP),))
        hidden = Dense(64, activation="sigmoid")(positions)
        hidden = Dense(64, activation="sigmoid")(hidden)
        out_from = Dense(64, activation="tanh")(hidden)
        out_to = Dense(64, activation="tanh")(hidden)

        model = keras.Model(inputs=[positions], outputs=[out_from, out_to])
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, brd):
        data = brd.piece_placement.flatten()[np.newaxis, ...]
        res = self._model.predict_on_batch(data)

        frm1 = res[0][0]
        frm2 = np.reshape(frm1, (-1, 8))
        tto1 = res[1][0]
        tto2 = np.reshape(tto1, (-1, 8))
        return frm2, tto2


class Player(object):
    board: Board
    nn: NN

    def __init__(self, brd, piece_index) -> None:
        super().__init__()
        self.piece_index = piece_index
        self.board = brd
        self.nn = NN()

    def get_move(self):
        weights_from, weights_to = self.nn.query(self.board)
        self._filter_positions(weights_from, 1 if self.piece_index else 0)
        self._filter_positions(weights_to, 0 if self.piece_index else 1)

        move = self._choose_move(weights_from, weights_to)
        return move

    def _choose_move(self, weights_from, weigts_to):
        # check 3-fold here
        return None

    def _filter_positions(self, weights, index):
        for rank in range(8):
            for col in range(8):
                cell = self.board.piece_placement[rank][col]
                idx = np.flatnonzero(cell)
                if not idx.size and index:
                    continue

                if not idx.size or idx[0] % 2 != index:
                    weights[rank][col] = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    board = Board()
    board.from_fen(STARTING_POSITION)

    white = Player(board, 0)
    black = Player(board, 1)

    while True:
        wmove = white.get_move()
        board.make_move(wmove)
        if not board.is_playable():
            break

        bmove = black.get_move()
        board.make_move(bmove)
        if not board.is_playable():
            break
