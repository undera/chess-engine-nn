import logging
import os
from random import shuffle

import numpy as np
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
        positions = layers.Input(shape=(8 * 8 * 12,), name="positions")  # 12 is len of PIECE_MAP

        reg = None
        activ_hidden = "sigmoid"
        kernel = 8 * 8
        hidden = layers.Dense((kernel * 10), activation=activ_hidden, kernel_regularizer=reg)(positions)
        hidden = layers.Dense((kernel * 8), activation=activ_hidden, kernel_regularizer=reg)(hidden)
        hidden = layers.Dense((kernel * 6), activation=activ_hidden, kernel_regularizer=reg)(hidden)
        hidden = layers.Dense((kernel * 4), activation=activ_hidden, kernel_regularizer=reg)(hidden)
        hidden = layers.Dense((kernel * 2), activation=activ_hidden, kernel_regularizer=reg)(hidden)

        out_from = layers.Dense(64, activation="sigmoid", name="from")(hidden)
        out_to = layers.Dense(64, activation="sigmoid", name="to")(hidden)

        model = Model(inputs=[positions, ], outputs=[out_from, out_to])
        model.compile(optimizer='adam',
                      loss=['categorical_crossentropy', 'categorical_crossentropy'],
                      loss_weights=[1.0, 1.0],
                      metrics=['categorical_accuracy']
                      )
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, fen):
        position = self._fen_to_array(fen).flatten()[np.newaxis, ...]
        frm, tto = self._model.predict_on_batch([position])

        return np.reshape(frm[0], (-1, 8)), np.reshape(tto[0], (-1, 8))

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

        out_from = np.full((batch_len, 64,), 0.0)
        out_to = np.full((batch_len, 64,), 0.0)
        outputs = [out_from, out_to]

        batch_n = 0
        while data:
            rec = data.pop(0)

            inputs_pos[batch_n] = self._fen_to_array(rec['fen']).flatten()

            score = rec['score']
            if not score or not rec['result']:
                continue

            out_from[batch_n][rec['move'].from_square] = score * rec['result']
            out_to[batch_n][rec['move'].to_square] = score * rec['result']

            # self._fill_eval(batch_n, out_evalb, rec['before'])
            # self._fill_eval(batch_n, out_evala, rec['after'])

            batch_n += 1

        res = self._model.fit(inputs, outputs, epochs=epochs, batch_size=128, verbose=2)
        # logging.debug("Trained: %s", res.history)
        # logging.debug("Trained: %s", res.history['loss'])
        # logging.debug("Scores: %.1f/%.1f", ns, ms)

    def _fill_eval(self, batch_n, out_evalb, rec):
        material, mobility, attacks, threats = rec
        out_evalb[batch_n][0] = material
        out_evalb[batch_n][1] = mobility
        out_evalb[batch_n][2] = attacks
        out_evalb[batch_n][3] = threats
