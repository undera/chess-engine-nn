import logging
import os
from collections import Counter

import numpy as np
from chess import PAWN, WHITE
from keras import layers, Model, models
from keras.utils import plot_model

PIECE_MAP = "PpNnBbRrQqKk"
PIECE_MOBILITY = {
    "P": 1,
    "N": 8,
    "B": 13,
    "R": 14,
    "Q": 27,
    "K": 64,
}


class NN(object):
    def __init__(self, filename) -> None:
        super().__init__()
        if os.path.exists(filename):
            self._model = models.load_model(filename)
        else:
            self._model = self._get_nn()
        self._model.summary(print_fn=logging.debug)

    def save(self, filename):
        self._model.save(filename, overwrite=True)

    def _get_nn(self):
        positions = layers.Input(shape=(8 * 8 * 12,))  # 12 is len of PIECE_MAP

        hidden = layers.Dense((8 * 8 * 10), activation="sigmoid")(positions)
        hidden = layers.Dense((8 * 8 * 8), activation="sigmoid")(hidden)
        hidden = layers.Dense((8 * 8 * 6), activation="sigmoid")(hidden)
        hidden = layers.Dense((8 * 8 * 4), activation="sigmoid")(hidden)
        hidden = layers.Dense((8 * 8 * 2), activation="sigmoid")(hidden)

        out_from = layers.Dense(64, activation="sigmoid")(hidden)
        out_to = layers.Dense(64, activation="sigmoid")(hidden)

        model = Model(inputs=[positions, ], outputs=[out_from, out_to])
        model.compile(optimizer='nadam',
                      loss='mse',
                      metrics=['accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, fen):
        position = self._fen_to_array(fen).flatten()[np.newaxis, ...]
        res = self._model.predict_on_batch([position])

        frm1 = res[0][0]
        frm2 = np.reshape(frm1, (-1, 8))
        tto1 = res[1][0]
        tto2 = np.reshape(tto1, (-1, 8))
        return frm2, tto2

    def _fen_to_array(self, fen):
        piece_placement = np.full((8, 8, 12), 0)  # rank, col, piece kind

        placement = fen
        rankn = 8
        for rank in placement.split('/'):
            rankn -= 1
            coln = 0
            for col in rank:
                try:
                    coln += int(col)
                except:
                    cell = piece_placement[rankn][coln]
                    cell[PIECE_MAP.index(col)] = 1
                    coln += 1

            assert coln == 8
        assert rankn == 0

        return piece_placement

    def learn(self, moves):
        filtered = True
        batch_len = len(self._learning_data)
        if not batch_len:
            logging.warning("No changes to learn from!")
            filtered = False
            batch_len = len(self._learning_data)
        inputs_pos = np.full((batch_len, 8 * 8 * 12), 0)
        inputs = inputs_pos

        out_from = np.full((batch_len, 64,), 0)
        out_to = np.full((batch_len, 64,), 1)
        outputs = [out_from, out_to]

        batch_n = 0
        ms = 0
        ns = 0
        prev_score = 0
        while self._learning_data:
            rec = self._learning_data.pop(0)
            if not self._learning_data and not rec['score'] and result:
                rec['score'] = result
                prev_score = 0

            if not rec['score'] and filtered:
                continue

            inputs_pos[batch_n] = self._fen_to_array(rec['board']).flatten()

            score = rec['score'] - prev_score
            ms = max(ms, score)
            ns = min(ns, score)
            score = score / 8.0
            if abs(score) > 1.0:
                score = np.sign(score)

            out_from[batch_n][rec['move'].from_square] = 1
            out_to[batch_n][rec['move'].to_square] = 0
            batch_n += 1
            prev_score = rec['score']

        res = self._model.fit(inputs, outputs, batch_size=batch_len, epochs=1, verbose=False)
        # logging.debug("Trained: %s", [res.history[key] for key in res.history if key.endswith("_acc")])
        # logging.debug("Trained: %s", res.history['loss'])
        # logging.debug("Scores: %.1f/%.1f", ns, ms)
        return prev_score

    def record_for_learning(self, board, selected_move):
        halfmove_score = board.halfmove_clock / 100.0
        self._learning_data.append({
            'board': board.board_fen(),
            'move': selected_move,
            'halfmove': halfmove_score,
            'fullmove': board.fullmove_number,
            'score': 0,
            'material': 0,
            'mobility': 0,
        })

    def after_our_move(self, board):
        self._opps_mobility = self._get_mobility(board)

    def after_their_move(self, board, side_coef):
        """
        :type board: chess.Board
        """
        if not self._learning_data:  # first move
            return

        prev = self._learning_data[-1]
        prev['material'] = self._get_material_balance(board)
        our_mobility = self._get_mobility(board)
        prev['mobility'] = our_mobility - self._opps_mobility
        score = prev['mobility'] / 10.0 + side_coef * prev['material']
        # logging.debug("Move of %s: %s %s", side_coef, prev['move'].san, score)
        prev['score'] = score

