import logging
import os

import numpy as np
from keras import layers, Model, models
from keras.layers import concatenate
from keras.utils import plot_model

PIECE_MAP = "PpNnBbRrQqKk"


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
        regulation = layers.Input(shape=(2,))  # 12 is len of PIECE_MAP
        inputs = concatenate([positions, regulation])

        hidden = layers.Dense(64, activation="relu")(inputs)
        hidden = layers.Dense(64, activation="relu")(hidden)

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

    def learn(self, score, batch):
        inputs_pos = np.full((len(batch), 8 * 8 * 12), 0)
        inputs_regul = np.full((len(batch), 2,), 0)
        inputs = [inputs_pos, inputs_regul]

        out_from = np.full((len(batch), 64,), 0)
        out_to = np.full((len(batch), 64,), 0)
        outputs = [out_from, out_to]

        batchNo = 0
        for fen, move, halfmove_score, fullmove in batch:
            inputs_pos[batchNo] = self.piece_placement_map(fen).flatten()
            inputs_regul[batchNo][0] = halfmove_score
            inputs_regul[batchNo][1] = fullmove

            out_from[batchNo][move.from_square] = score
            out_to[batchNo][move.to_square] = score
            batchNo += 1

        res = self._model.fit(inputs, outputs, batch_size=len(batch), epochs=1, verbose=False)
        # logging.debug("Trained: %s", [res.history[key] for key in res.history if key.endswith("_acc")])
        # logging.debug("Trained: %s", res.history['loss'])
