import logging
import os
from random import shuffle

import numpy as np
from keras import layers, Model, models
from keras.regularizers import l2
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

        reg = l2(0.01)
        hidden = layers.Dense((8 * 8 * 6), activation="tanh", kernel_regularizer=reg)(positions)
        hidden = layers.Dropout(0.05)(hidden)
        hidden = layers.Dense((8 * 8 * 4), activation="tanh", kernel_regularizer=reg)(hidden)
        out_evalb = layers.Dense(4, activation="tanh", name="evalb")(hidden)
        hidden = layers.Dropout(0.05)(hidden)
        hidden = layers.Dense((8 * 8 * 2), activation="tanh", kernel_regularizer=reg)(hidden)
        hidden = layers.Dropout(0.05)(hidden)
        hidden = layers.Dense((8 * 8 * 1), activation="relu", kernel_regularizer=reg)(hidden)

        out_from = layers.Dense(64, activation="sigmoid", name="from")(hidden)
        out_to = layers.Dense(64, activation="sigmoid", name="to")(hidden)
        out_evala = layers.Dense(4, activation="tanh", name="evala")(hidden)

        model = Model(inputs=[positions, ], outputs=[out_from, out_to, out_evalb, out_evala])
        model.compile(optimizer='adam',
                      loss=['binary_crossentropy', 'binary_crossentropy', 'mse', 'mse'],
                      loss_weights=[1.0, 1.0, 1.0, 1.0],
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

    def learn(self, data, epochs):
        shuffle(data)
        batch_len = len(data)
        inputs_pos = np.full((batch_len, 8 * 8 * 12), 0)
        inputs = inputs_pos

        out_from = np.full((batch_len, 64,), 0)
        out_to = np.full((batch_len, 64,), 0)
        out_evalb = np.full((batch_len, 4,), 0)
        out_evala = np.full((batch_len, 4,), 0)
        outputs = [out_from, out_to, out_evalb, out_evala]

        batch_n = 0
        while data:
            rec = data.pop(0)

            inputs_pos[batch_n] = self._fen_to_array(rec['fen']).flatten()

            score = self._get_score(rec["before"], rec["after"])

            out_from[batch_n][rec['move'].from_square] = score * rec['score']
            out_to[batch_n][rec['move'].to_square] = score * rec['score']

            self._fill_eval(batch_n, out_evalb, rec['before'])
            self._fill_eval(batch_n, out_evala, rec['after'])

            batch_n += 1

        res = self._model.fit(inputs, outputs, epochs=epochs, verbose=2)
        # logging.debug("Trained: %s", [res.history[key] for key in res.history if key.endswith("_acc")])
        # logging.debug("Trained: %s", res.history['loss'])
        # logging.debug("Scores: %.1f/%.1f", ns, ms)

    def _fill_eval(self, batch_n, out_evalb, rec):
        material, mobility, attacks, threats = rec
        out_evalb[batch_n][0] = material
        out_evalb[batch_n][1] = mobility
        out_evalb[batch_n][2] = attacks
        out_evalb[batch_n][3] = threats

    def _get_score(self, before, after):
        if after[0] > before[0]:
            return 1
        elif after[0] < before[0]:
            return 0

        att_balance_b = before[2] - before[3]
        att_balance_a = after[2] - after[3]
        if att_balance_a > att_balance_b:
            return 0.75
        elif att_balance_a < att_balance_b:
            return 0

        if after[1] > before[1]:
            return 0.5
        elif after[1] < before[1]:
            return 0

        return 0
