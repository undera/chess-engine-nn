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
        logging.info("New move: %s", move)
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
        self._filter_positions(weights_from, 1 if self.piece_index else 0, False)
        self._filter_positions(weights_to, 0 if self.piece_index else 1, True)

        rev_from = self._reverse_index(weights_from)
        rev_to = self._reverse_index(weights_to)

        move = self._choose_move(rev_from, rev_to)
        # TODO: check 3-fold here
        return move

    def _choose_move(self, rev_from, rev_to):
        possible_moves = []
        for ffrom in rev_from:
            piece_class = PIECE_MAP[ffrom[3]].upper()
            for tto in rev_to:
                if self._is_valid_move(piece_class, ffrom, tto):
                    possible_moves.append((ffrom[0] * tto[0], ffrom, tto))

        possible_moves.sort(key=lambda x: x[0], reverse=True)
        # check for exposing king to check
        return possible_moves[0] if possible_moves else None

    def _is_valid_move(self, piece_class, src, dest):
        _, src_r, src_c, src_p = src
        _, dst_r, dst_c, dst_p = dest
        if piece_class == 'P':
            direction = -1 if self.piece_index else 1

            if src_c == dst_c:
                if src_r + direction == dst_r and dst_p is None:
                    # regular move
                    return True
                elif src_r + direction * 2 == dst_r and dst_p is None \
                        and self._piece_at(src_r + direction, src_c) is None:
                    # first move
                    return True
            elif abs(src_c - dst_c) == 1 and src_r + direction == dst_r and dst_p is not None:
                # capture is special
                return True
            # TODO: en passant
        elif piece_class == 'N':
            if (abs(src_r - dst_r) == 2 and abs(src_c - dst_c) == 1) or \
                    (abs(src_r - dst_r) == 1 and abs(src_c - dst_c) == 2):
                return True
        elif piece_class == 'B':
            pass
        elif piece_class == 'R':
            pass
        elif piece_class == 'Q':
            pass
        elif piece_class == 'K':
            pass
        else:
            raise ValueError()
        return False

    def _filter_positions(self, weights, index, allow_empty):
        for rank in range(8):
            for col in range(8):
                cell = self.board.piece_placement[rank][col]
                idx = np.flatnonzero(cell)
                if not idx.size and allow_empty:
                    continue

                if not idx.size or idx[0] % 2 != index:
                    weights[rank][col] = None

    def _reverse_index(self, weights):
        idx = []
        for rank in range(8):
            for col in range(8):
                if not np.isnan(weights[rank][col]):
                    idx.append((weights[rank][col], rank, col, self._piece_at(rank, col)))

        # idx.sort(key=lambda x: x[0], reverse=True)
        return idx

    def _piece_at(self, rank, col):
        cell = self.board.piece_placement[rank][col]
        piece_idx = np.flatnonzero(cell)
        return piece_idx[0] if piece_idx.size else None

