import logging
import os
import random
import time
from collections import Counter
from typing import List

import numpy as np
from keras import layers, Model, models
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from keras.utils import plot_model

PIECE_MAP = "PpNnBbRrQqKk"


class NN(object):
    activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
    optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam

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

        reg = l2(0.0001)
        kernel = 8 * 8
        hidden = layers.Dense(kernel, activation=self.activ_hidden, kernel_regularizer=reg)(positions)
        hidden = layers.Dense(kernel, activation=self.activ_hidden, kernel_regularizer=reg)(hidden)

        out_from = layers.Dense(64, activation="softmax", name="from")(hidden)
        out_to = layers.Dense(64, activation="softmax", name="to")(hidden)

        model = Model(inputs=[positions, ], outputs=[out_from, out_to])
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      loss_weights=[0.1, 1.0],
                      metrics=['categorical_accuracy'])
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
        data: List[MoveRecord] = list(filter(lambda x: x.get_score() > 0.0, data))
        random.shuffle(data)

        score_dist = Counter([x.get_score() for x in data])
        logging.info("Scores: %s", ["%.1f: %.2f" % (x, score_dist[x] / float(len(data))) for x in score_dist])
        piece_dist = Counter([x.piece.symbol().upper() for x in data])
        piece_dist = {x: piece_dist[x] / float(len(data)) for x in piece_dist}
        logging.info("Pieces: %s", {x: round(y * 100) for x, y in piece_dist.items()})

        batch_len = len(data)
        inputs_pos = np.full((batch_len, 8 * 8 * 12), 0)
        inputs = inputs_pos

        sample_weights = [np.full((batch_len,), 1.0), np.full((batch_len,), 1.0)]
        out_from = np.full((batch_len, 64,), 0.0)
        out_to = np.full((batch_len, 64,), 0.0)
        outputs = [out_from, out_to]

        batch_n = 0
        for rec in data:
            sample_weights[0][batch_n] = 1.0 - piece_dist[rec.piece.symbol().upper()]
            sample_weights[1][batch_n] = 1.0 - piece_dist[rec.piece.symbol().upper()]

            inputs_pos[batch_n] = self._fen_to_array(rec.fen).flatten()

            out_from[batch_n] = np.full((64,), 0.0)
            out_to[batch_n] = np.full((64,), 0.0)

            score = rec.get_score()
            out_from[batch_n][rec.from_square] = score
            out_to[batch_n][rec.to_square] = score

            # self._fill_eval(batch_n, out_evalb, rec['before'])
            # self._fill_eval(batch_n, out_evala, rec['after'])

            batch_n += 1

        res = self._model.fit(inputs, outputs, sample_weight=sample_weights,
                              validation_split=0.1, shuffle=True,
                              callbacks=[TensorBoard('/tmp/tensorboard/%d' % time.time())], verbose=0,
                              epochs=epochs, batch_size=128, )
        logging.debug("Trained: %s", res.history)

    def _fill_eval(self, batch_n, out_evalb, rec):
        material, mobility, attacks, threats = rec
        out_evalb[batch_n][0] = material
        out_evalb[batch_n][1] = mobility
        out_evalb[batch_n][2] = attacks
        out_evalb[batch_n][3] = threats
