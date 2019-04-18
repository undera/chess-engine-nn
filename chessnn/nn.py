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
        kernel = 8 * 8
        activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
        activ_out = "softmax"  # linear relu elu sigmoid tanh softmax
        optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam

        def _residual(inp, size):
            out = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(inp)
            return concatenate([inp, out])

        def _branch(layer, repeats, name):
            for _ in range(repeats):
                layer = _residual(layer, kernel)

            odense = layers.Dense(64, activation=activ_out)(layer)
            omatrix = layers.Reshape((8, 8), name=name)(odense)
            return odense, omatrix

        position = layers.Input(shape=(8, 8, 2, len(PIECE_TYPES),), name="position")
        iflat = layers.Flatten()(position)

        pmoves, out_pmoves = _branch(iflat, 4, "possible_moves")
        attacks, out_attacks = _branch(iflat, 4, "attacks")
        defences, out_defences = _branch(iflat, 4, "defences")
        threats, out_threats = _branch(iflat, 4, "threats")
        threatened, out_threatened = _branch(iflat, 4, "threatened")

        bfrom = concatenate([iflat, defences, threatened])
        bfrom, out_from = _branch(bfrom, 8, "main_from")

        bto = concatenate([iflat, pmoves, attacks, threats])
        bto, out_to = _branch(bto, 8, "main_to")

        beval = concatenate([iflat, pmoves, attacks, defences, threats, threatened])
        for x in range(4):
            beval = _residual(beval, kernel)
        oeval = layers.Dense(1, activation="sigmoid", name="eval")(beval)

        outputs = [oeval, out_from, out_to, out_pmoves, out_attacks, out_defences, out_threats, out_threatened]
        model = Model(inputs=[position, ], outputs=outputs)
        model.compile(optimizer=optimizer,
                      loss=["binary_crossentropy", ] + ['categorical_crossentropy'] * 7,
                      loss_weights=[1.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      metrics=['categorical_accuracy', "accuracy"])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, position):
        res = self._model.predict_on_batch([position])

        return (x[0] for x in res)

    def learn(self, data, epochs, force_score=None):
        if not data:
            logging.warning("No data to train on")
            return

        batch_len = len(data)
        inputs_pos = np.full((batch_len, 8, 8, 2, len(PIECE_TYPES)), 0)
        inputs = inputs_pos

        out_from = np.full((batch_len, 8, 8), 0.0)
        out_to = np.full((batch_len, 8, 8), 0.0)
        pmoves = np.full((batch_len, 8, 8), 0.0)
        attacks = np.full((batch_len, 8, 8), 0.0)
        defences = np.full((batch_len, 8, 8), 0.0)
        threats = np.full((batch_len, 8, 8), 0.0)
        threatened = np.full((batch_len, 8, 8), 0.0)
        evals = np.full((batch_len, 1), 0.0)

        outputs = [evals, out_from, out_to, pmoves, attacks, defences, threats, threatened]

        batch_n = 0
        for rec in data:
            assert isinstance(rec, MoveRecord)
            score = rec.get_eval() if force_score is None else force_score
            assert score is not None

            evals[batch_n][0] = score
            inputs_pos[batch_n] = rec.position

            out_from[batch_n][square_file(rec.from_square)][square_rank(rec.from_square)] = 1
            out_to[batch_n][square_file(rec.to_square)][square_rank(rec.to_square)] = 1

            pmoves[batch_n] = np.reshape(rec.possible_moves, (-1, 8))
            attacks[batch_n] = np.reshape(rec.attacked, (-1, 8))
            defences[batch_n] = np.reshape(rec.defended, (-1, 8))
            threats[batch_n] = np.reshape(rec.threats, (-1, 8))
            threatened[batch_n] = np.reshape(rec.threatened, (-1, 8))

            batch_n += 1

        cbs = [TensorBoard('/tmp/tensorboard/%d' % time.time())] if epochs > 1 else []
        res = self._model.fit(inputs, outputs,
                              # validation_split=0.1, shuffle=True,
                              callbacks=cbs, verbose=2,
                              epochs=epochs, batch_size=128, )
        logging.debug("Trained: %s", res.history)
