import logging

import keras
import numpy as np
from keras import Input
from keras.layers import Dense, Activation
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

    def playable(self):
        return True

    def make_move(self, move):
        # log the move
        pass

    def from_fen(self, fen):
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
        out_from = Dense(64, activation="sigmoid")(hidden)
        out_to = Dense(64, activation="sigmoid")(hidden)

        model = keras.Model(inputs=[positions], outputs=[out_from, out_to])
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, brd):
        data = brd.piece_placement.flatten()[np.newaxis, ...]
        res = self._model.predict_on_batch(data)
        return res[0][0], res[1][0]


class Player(object):
    board: Board
    nn: NN

    def __init__(self, brd, piece_index) -> None:
        super().__init__()
        self.piece_index = piece_index
        self.board = brd
        self.nn = NN()

    def get_move(self):
        weights_from, weigts_to = self.nn.query(self.board)

        move = self._choose_move(weights_from, weigts_to)
        return move

    def _choose_move(self, weights_from, weigts_to):
        # check 3-fold here
        return None


if __name__ == "__main__":
    board = Board()
    board.from_fen(STARTING_POSITION)

    white = Player(board, 0)
    black = Player(board, 1)

    while board.playable():
        wmove = white.get_move()
        board.make_move(wmove)

        bmove = black.get_move()
        board.make_move(bmove)
