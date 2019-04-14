import logging
import os
import time

import numpy as np
from chess import PIECE_TYPES, square_file, square_rank
from keras import layers, Model, models
from keras.callbacks import TensorBoard
from keras.layers import concatenate
from keras.utils import plot_model

from chessnn import MoveRecord


class NN(object):
    activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
    optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam

    def __init__(self, filename) -> None:
        super().__init__()
        if os.path.exists(filename):
            logging.info("Loading model from: %s", filename)
            self._model = models.load_model(filename)
        else:
            logging.info("Starting with clean model")
            self._model = self._get_nn()
        self._model.summary(print_fn=logging.debug)

    def save(self, filename):
        logging.info("Saving model to: %s", filename)
        self._model.save(filename, overwrite=True)

    def _get_nn(self):
        reg = None  # l2(0.0001)
        kernel = 8 * 8 * 2
        activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
        activ_out = "softmax"  # linear relu elu sigmoid tanh softmax
        optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam

        def _residual(inp):
            out = layers.Dense(kernel, activation=activ_hidden)(inp)
            return concatenate([inp, out])

        position = layers.Input(shape=(8, 8, 2, len(PIECE_TYPES),), name="position")
        hidden = layers.Flatten()(position)
        hidden = _residual(hidden)
        hidden = _residual(hidden)
        hidden = _residual(hidden)
        hidden = _residual(hidden)

        pmoves = layers.Dense(64, activation=activ_out, kernel_regularizer=reg)(hidden)
        possible_moves = layers.Reshape((8, 8), name="possible_moves")(pmoves)
        hidden = concatenate([hidden, pmoves])

        hidden = _residual(hidden)
        hidden = _residual(hidden)

        out_from = layers.Dense(64, activation=activ_out, kernel_regularizer=reg)(hidden)
        out_from = layers.Reshape((8, 8), name="from")(out_from)
        out_to = layers.Dense(64, activation=activ_out, kernel_regularizer=reg)(hidden)
        out_to = layers.Reshape((8, 8), name="to")(out_to)

        model = Model(inputs=[position, ], outputs=[possible_moves, out_from, out_to])
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      loss_weights=[1.0, 1.0, 1.0],
                      metrics=['categorical_accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, position):
        possible_moves, frm, tto = self._model.predict_on_batch([position])

        return frm[0], tto[0], possible_moves[0]

    def learn(self, data, epochs, force_score=None):
        # data: List[MoveRecord] = list(filter(lambda x: x.get_score() > 0.0, data))

        batch_len = len(data)
        inputs_pos = np.full((batch_len, 8, 8, 2, len(PIECE_TYPES)), 0)
        inputs = inputs_pos

        possible_moves = np.full((batch_len, 8, 8), 0.0)
        out_from = np.full((batch_len, 8, 8), 0.0)
        out_to = np.full((batch_len, 8, 8), 0.0)
        outputs = [possible_moves, out_from, out_to]

        batch_n = 0
        rec: MoveRecord
        for rec in data:
            score = rec.get_score() if force_score is None else force_score
            assert score is not None

            inputs_pos[batch_n] = rec.position

            possible_moves[batch_n] = np.reshape(rec.possible_moves, (-1, 8))
            out_from[batch_n][square_file(rec.from_square)][square_rank(rec.from_square)] = score
            out_to[batch_n][square_file(rec.to_square)][square_rank(rec.to_square)] = score

            # self._fill_eval(batch_n, out_evalb, rec['before'])
            # self._fill_eval(batch_n, out_evala, rec['after'])

            batch_n += 1

        cbs = [TensorBoard('/tmp/tensorboard/%d' % time.time())] if epochs > 1 else []
        res = self._model.fit(inputs, outputs,
                              validation_split=0.1, shuffle=True,
                              callbacks=cbs, verbose=2,
                              epochs=epochs, batch_size=128, )
        logging.debug("Trained: %s", res.history)

    def _fill_eval(self, batch_n, out_evalb, rec):
        material, mobility, attacks, threats = rec
        out_evalb[batch_n][0] = material
        out_evalb[batch_n][1] = mobility
        out_evalb[batch_n][2] = attacks
        out_evalb[batch_n][3] = threats
