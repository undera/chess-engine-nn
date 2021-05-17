import logging
import os
import time
from abc import abstractmethod
from operator import itemgetter

import chess
import numpy as np
from chess import PIECE_TYPES
from keras import models, callbacks, layers, regularizers
from keras.utils.vis_utils import plot_model

from chessnn import MoveRecord, MOVES_MAP

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NN(object):
    _model: models.Model

    def __init__(self, filename=None) -> None:
        super().__init__()
        self.loaded = False
        self._train_acc_threshold = 0.9
        self._validate_acc_threshold = 0.9
        if filename and os.path.exists(filename):
            logging.info("Loading model from: %s", filename)
            self._model = models.load_model(filename)
            self.loaded = True
        else:
            logging.info("Starting with clean model")
            self._model = self._get_nn()
            self._model.summary(print_fn=logging.info)
            plot_model(self._model, to_file=os.path.join(os.path.dirname(__file__), '..', 'model.png'),
                       show_shapes=True)

    def save(self, filename):
        logging.info("Saving model to: %s", filename)
        self._model.save(filename, overwrite=True)

    def inference(self, data):
        inputs, outputs = self._data_to_training_set(data, True)
        res = self._model.predict_on_batch(inputs)
        return [x[0] for x in res]

    def train(self, data, epochs, validation_data=None):
        self.loaded = True
        logging.info("Preparing training set...")
        inputs, outputs = self._data_to_training_set(data, False)

        logging.info("Starting to learn...")
        cbpath = '/tmp/tensorboard/%d' % (time.time() if epochs > 1 else 0)
        cbs = [callbacks.TensorBoard(cbpath, write_graph=False, profile_batch=0)]
        res = self._model.fit(inputs, outputs,  # sample_weight=np.array(sample_weights),
                              validation_split=0.1 if (validation_data is None and epochs > 1) else 0.0, shuffle=True,
                              callbacks=cbs, verbose=2 if epochs > 1 else 0,
                              epochs=epochs)
        logging.info("Trained: %s", {x: y[-1] for x, y in res.history.items()})

        if validation_data is not None:
            self.validate(validation_data)

    def validate(self, data):
        logging.info("Preparing validation set...")
        inputs, outputs = self._data_to_training_set(data, False)

        logging.info("Starting to validate...")
        res = self._model.evaluate(inputs, outputs)
        logging.info("Validation loss and KPIs: %s", res)
        msg = "Validation accuracy is too low: %.3f < %s" % (res[1], self._validate_acc_threshold)
        assert res[1] >= self._validate_acc_threshold, msg

    @abstractmethod
    def _get_nn(self):
        pass

    @abstractmethod
    def _data_to_training_set(self, data, is_inference=False):
        pass


reg = None  # regularizers.l2(0.001)
activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
optimizer = "rmsprop"  # sgd rmsprop adagrad adadelta adamax adam nadam


class NNChess(NN):
    def _get_nn(self):
        pos_shape = (8, 8, len(PIECE_TYPES) * 2)
        position = layers.Input(shape=pos_shape, name="position")
        flags = layers.Input(shape=(3,), name="flags")
        pos_analyzed = self.__nn_conv(position)

        out_possible = layers.Dense(len(MOVES_MAP), activation="sigmoid", kernel_regularizer=reg, name="possible")(
            pos_analyzed)
        out_attacked = layers.Dense(64, activation="sigmoid", kernel_regularizer=reg, name="attacked")(pos_analyzed)
        out_defended = layers.Dense(64, activation="sigmoid", kernel_regularizer=reg, name="defended")(pos_analyzed)

        conc = layers.concatenate([out_possible, out_attacked, out_defended, layers.Flatten()(position), flags])
        main = layers.Dense(128, activation='relu', kernel_regularizer=reg)(conc)
        main = layers.Dense(128, activation='relu', kernel_regularizer=reg)(main)
        out_moves = layers.Dense(len(MOVES_MAP), activation="softmax", name="moves")(main)
        out_eval = layers.Dense(1, activation="sigmoid", name="eval")(main)

        model = models.Model(inputs=[position, flags],
                             outputs=[out_moves, out_eval, out_possible, out_attacked, out_defended])
        model.compile(optimizer=optimizer,
                      loss=["categorical_crossentropy",
                            "binary_crossentropy", "binary_crossentropy", "binary_crossentropy",
                            "binary_crossentropy", ],
                      loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
                      metrics=['binary_accuracy', 'categorical_accuracy', "accuracy"])
        return model

    def __nn_simple(self, position):
        flat = layers.Flatten()(position)
        main1 = layers.Dense(128, activation=activ_hidden, kernel_regularizer=reg)(flat)
        conc1 = layers.concatenate([flat, main1])
        main2 = layers.Dense(64, activation=activ_hidden, kernel_regularizer=reg)(conc1)
        conc2 = layers.concatenate([flat, main2])
        main3 = layers.Dense(128, activation=activ_hidden, kernel_regularizer=reg)(conc2)
        return main3

    def __nn_residual(self, position):
        # flags = layers.Input(shape=(1,), name="flags")
        main = layers.Flatten()(position)

        def _residual(inp, size):
            # out = layers.Dropout(rate=0.05)(inp)
            inp = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(inp)
            out = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(inp)
            out = layers.merge.multiply([inp, out])
            # out = layers.Dense(size, activation=activ_hidden, kernel_regularizer=reg)(out)
            return out

        branch = main
        for _ in range(1, 4):
            branch = _residual(branch, 8 * 8 * _)

        return branch

    def __nn_conv(self, position):
        conv31 = layers.Conv2D(8, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(position)
        conv32 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv31)
        conv33 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv32)
        flat3 = layers.Flatten()(conv33)

        conv41 = layers.Conv2D(8, kernel_size=(4, 4), activation="relu", kernel_regularizer=reg)(position)
        conv42 = layers.Conv2D(16, kernel_size=(4, 4), activation="relu", kernel_regularizer=reg)(conv41)
        flat4 = layers.Flatten()(conv42)

        conv51 = layers.Conv2D(8, kernel_size=(5, 5), activation="relu", kernel_regularizer=reg)(position)
        conv52 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv51)
        flat5 = layers.Flatten()(conv52)

        conv61 = layers.Conv2D(8, kernel_size=(6, 6), activation="relu", kernel_regularizer=reg)(position)
        # conv62 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv61)
        flat6 = layers.Flatten()(conv61)

        conv71 = layers.Conv2D(8, kernel_size=(7, 7), activation="relu", kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv71)
        flat7 = layers.Flatten()(conv71)

        conv81 = layers.Conv2D(8, kernel_size=(8, 8), activation="relu", kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv71)
        flat8 = layers.Flatten()(conv81)

        conc = layers.concatenate([flat3, flat4, flat5, flat6, flat7, flat8])
        return conc

    def _data_to_training_set(self, data, is_inference=False):
        batch_len = len(data)

        inputs_pos = np.full((batch_len, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        inputs_flags = np.full((batch_len, 3), 0.0)
        out_moves = np.full((batch_len, len(MOVES_MAP)), 0.0)
        out_possible = np.full((batch_len, len(MOVES_MAP)), 0.0)
        out_attacks = np.full((batch_len, 64), 0.0)
        out_threats = np.full((batch_len, 64), 0.0)
        out_evals = np.full((batch_len, 1), 0.0)

        batch_n = 0
        for moverec in data:
            assert isinstance(moverec, MoveRecord)

            evl = moverec.eval

            out_evals[batch_n][0] = evl
            inputs_pos[batch_n] = moverec.position

            inputs_flags[batch_n][0] = 1.0 / moverec.full_move
            inputs_flags[batch_n][1] = 1.0 / (moverec.fifty_progress + 1)
            inputs_flags[batch_n][2] = moverec.fifty_progress / 100.0

            out_moves[batch_n] = np.full((len(MOVES_MAP, )), 0.0 if evl else 1.0 / (len(MOVES_MAP) - 1))
            out_moves[batch_n][moverec.get_move_num()] = evl
            out_possible[batch_n] = moverec.possible
            out_attacks[batch_n] = moverec.attacked
            out_threats[batch_n] = moverec.defended

            batch_n += 1

        return [inputs_pos, inputs_flags], [out_moves, out_evals, out_possible, out_attacks, out_threats]

    def inference(self, data):
        inference = super().inference(data)
        return [self._moves_iter(inference[0]), inference[1]] + [inference[2:] + [inference[0]]]

    def _moves_iter(self, scores):
        for idx, score in sorted(np.ndenumerate(scores), key=itemgetter(1), reverse=True):
            idx = idx[0]
            move = chess.Move(MOVES_MAP[idx][0], MOVES_MAP[idx][1])
            yield move
