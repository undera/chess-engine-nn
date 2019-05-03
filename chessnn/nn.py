import logging
import os
import time

import keras
import numpy as np
from chess import PIECE_TYPES, square_file, square_rank
from keras import layers, Model, models
from keras.callbacks import TensorBoard
from keras.layers import merge
from keras.utils import plot_model

from chessnn import MoveRecord


class NN(object):
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
        reg = None  # keras.regularizers.l2(0.00001)
        kernel = 8 * 8
        activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
        activ_out = "softmax"  # linear relu elu sigmoid tanh softmax
        optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam

        def _residual(inp, size):
            # out = layers.Dropout(rate=0.05)(inp)
            inp = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(inp)
            out = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(inp)
            out = merge.multiply([inp, out])
            # out = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(out)
            return out

        def _branch(layer, wfrom, wto, name):
            for _ in range(wfrom, wto - 2, -2):
                layer = _residual(layer, kernel * _)

            odense = layers.Dense(64, activation="softmax" if name in ("from", "to") else activ_out)(layer)
            omatrix = layers.Reshape((8, 8), name=name)(odense)
            return odense, omatrix

        position = layers.Input(shape=(2, 8, 8, len(PIECE_TYPES),), name="position")

        iflat = layers.Flatten()(position)
        # iflat = layers.Dense(kernel, activation=activ_hidden, kernel_regularizer=reg)(iflat)

        pmoves, out_pmoves = _branch(iflat, 10, 8, "possible_moves")
        attacks, out_attacks = _branch(iflat, 10, 8, "attacks")
        defences, out_defences = _branch(iflat, 10, 8, "defences")
        threats, out_threats = _branch(iflat, 10, 8, "threats")
        threatened, out_threatened = _branch(iflat, 10, 8, "threatened")

        main = layers.concatenate([iflat, pmoves, attacks, defences, threats, threatened])
        ffrom = main  # layers.concatenate([iflat, defences, threatened])
        fto = main  # layers.concatenate([iflat, pmoves, attacks, threats, ])

        ffrom, out_from = _branch(ffrom, 8, 2, "from")
        fto, out_to = _branch(fto, 8, 2, "to")

        beval = main  # layers.concatenate([iflat, pmoves, attacks, defences, threats, threatened])

        for x in range(2):
            beval = _residual(beval, kernel)
        oeval = layers.Dense(2, activation=activ_out, name="eval")(beval)

        outputs = [oeval, out_from, out_to, out_pmoves, out_attacks, out_defences, out_threats, out_threatened]
        model = Model(inputs=[position, ], outputs=outputs)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      loss_weights=[0.5, 1.0, 1.0] + [0.25] + [0.25] * 4,
                      metrics=['categorical_accuracy'])
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
        inputs_pos = np.full((batch_len, 2, 8, 8, len(PIECE_TYPES)), 0)
        inputs = inputs_pos

        evals = np.full((batch_len, 2), 0.0)
        out_from = np.full((batch_len, 8, 8), 0.0)
        out_to = np.full((batch_len, 8, 8), 0.0)

        pmoves = np.full((batch_len, 8, 8), 0.0)
        attacks = np.full((batch_len, 8, 8), 0.0)
        defences = np.full((batch_len, 8, 8), 0.0)
        threats = np.full((batch_len, 8, 8), 0.0)
        threatened = np.full((batch_len, 8, 8), 0.0)

        outputs = [evals, out_from, out_to] + [pmoves, attacks, defences, threats, threatened]

        batch_n = 0
        for rec in data:
            assert isinstance(rec, MoveRecord)
            score = rec.get_eval() if force_score is None else force_score
            assert score is not None

            evals[batch_n][0] = score
            evals[batch_n][1] = 1.0 - score
            inputs_pos[batch_n] = rec.position

            out_from[batch_n] = np.full((8, 8), 0.0 if score >= 0.5 else 1.0 / 64)
            out_to[batch_n] = np.full((8, 8), 0.0 if score >= 0.5 else 1.0 / 64)

            out_from[batch_n][square_file(rec.from_square)][square_rank(rec.from_square)] = 1 if score >= 0.5 else 0
            out_to[batch_n][square_file(rec.to_square)][square_rank(rec.to_square)] = 1 if score >= 0.5 else 0

            pmoves[batch_n] = rec.possible_moves
            attacks[batch_n] = rec.attacked
            defences[batch_n] = rec.defended
            threats[batch_n] = rec.threats
            threatened[batch_n] = rec.threatened

            batch_n += 1

        assert not np.isnan(inputs.flatten()).any()
        for out in outputs:
            assert not np.isnan(out.flatten()).any()

        cbs = [TensorBoard('/tmp/tensorboard/%d' % time.time())] if epochs > 1 else []
        res = self._model.fit(inputs, outputs,
                              # validation_split=0.1, shuffle=True,
                              callbacks=cbs, verbose=2,
                              epochs=epochs, batch_size=128, )
        logging.debug("Trained: %s", res.history)
