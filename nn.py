import logging
import os
from collections import Counter

import numpy as np
from chess import PAWN, WHITE
from keras import layers, Model, models
from keras.layers import concatenate
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
        self._learning_data = []
        self._opps_mobility = 0

    def save(self, filename):
        self._model.save(filename, overwrite=True)

    def _get_nn(self):
        positions = layers.Input(shape=(8 * 8 * 12,))  # 12 is len of PIECE_MAP
        regulation = layers.Input(shape=(2,))  # 12 is len of PIECE_MAP
        inputs = concatenate([positions, regulation])

        hidden = layers.Dense(64, activation="sigmoid")(inputs)
        hidden = layers.Dense(64, activation="sigmoid")(hidden)

        out_from = layers.Dense(64, activation="tanh")(hidden)
        out_to = layers.Dense(64, activation="tanh")(hidden)

        model = Model(inputs=[positions, regulation], outputs=[out_from, out_to])
        model.compile(optimizer='nadam',
                      loss='mse',
                      metrics=['accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, fen, fiftyturnscore, fullmove):
        position = self.piece_placement_map(fen).flatten()[np.newaxis, ...]
        regulations = np.zeros((2,))[np.newaxis, ...]
        regulations[0][0] = fiftyturnscore
        regulations[0][1] = fullmove
        if fiftyturnscore > 0.5:
            pass
        res = self._model.predict_on_batch([position, regulations])

        frm1 = res[0][0]
        frm2 = np.reshape(frm1, (-1, 8))
        tto1 = res[1][0]
        tto2 = np.reshape(tto1, (-1, 8))
        return frm2, tto2

    def piece_placement_map(self, fen):
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

    def learn(self, result, side_coef):
        filtered = True
        batch_len = len([x for x in self._learning_data if x['score']])
        if not batch_len:
            logging.warning("No changes to learn from!")
            filtered = False
            batch_len = len(self._learning_data)
        inputs_pos = np.full((batch_len, 8 * 8 * 12), 0)
        inputs_regul = np.full((batch_len, 2,), 0)
        inputs = [inputs_pos, inputs_regul]

        out_from = np.full((batch_len, 64,), 0)
        out_to = np.full((batch_len, 64,), 0)
        outputs = [out_from, out_to]

        batch_n = 0
        ms = 0
        ns = 0
        for rec in self._learning_data:
            if not rec['score'] and filtered:
                continue
            inputs_pos[batch_n] = self.piece_placement_map(rec['board']).flatten()
            inputs_regul[batch_n][0] = rec['halfmove']
            inputs_regul[batch_n][1] = rec['fullmove']

            ms = max(ms, rec['score'])
            ns = min(ns, rec['score'])
            score = rec['score'] / 64.0
            if abs(score) > 1.0:
                score = np.sign(score)
            out_from[batch_n][rec['move'].from_square] = score
            out_to[batch_n][rec['move'].to_square] = score
            batch_n += 1

        res = self._model.fit(inputs, outputs, batch_size=batch_len, epochs=1, verbose=False)
        # logging.debug("Trained: %s", [res.history[key] for key in res.history if key.endswith("_acc")])
        # logging.debug("Trained: %s", res.history['loss'])
        # logging.debug("Scores: %.1f/%.1f", ns, ms)
        self._learning_data.clear()

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
        if len(self._learning_data) > 1:
            material_change = prev['material'] - self._learning_data[-2]['material']
            mobility_change = prev['mobility'] - self._learning_data[-2]['mobility']
        else:
            material_change = 0
            mobility_change = prev['mobility']
        score = mobility_change / 10.0 + side_coef * material_change
        # logging.debug("Move of %s: %s %s", side_coef, prev['move'].san, score)
        prev['score'] = score

    def _get_material_balance(self, board):
        fen = board.board_fen()
        chars = Counter(fen)
        score = 0
        for piece in PIECE_MOBILITY:
            if piece in chars:
                score += PIECE_MOBILITY[piece] * chars[piece]

            if piece.lower() in chars:
                score -= PIECE_MOBILITY[piece] * chars[piece.lower()]

        return score

    def _get_mobility(self, board):
        """

        :type board: chess.Board
        """
        score = 0
        moves = list(board.generate_legal_moves())
        for move in moves:
            src_piece = board.piece_at(move.from_square)
            if src_piece.piece_type == PAWN:
                if src_piece.color == WHITE:
                    score += 1 + 0.1 * (move.to_square // 8 - 8)
                else:
                    score += 1 + 0.1 * (8 - move.to_square // 8)
            else:
                score += 1

            dest_piece = board.piece_at(move.to_square)
            if dest_piece:  # attack bonus
                score += PIECE_MOBILITY[dest_piece.symbol().upper()]
        return score
