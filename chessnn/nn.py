import hashlib
import logging
import os
import tempfile
import time
from abc import abstractmethod
from operator import itemgetter

import chess
import numpy as np
import tensorflow
from chess import PIECE_TYPES
from keras import models, callbacks, layers, regularizers
from keras.utils.vis_utils import plot_model
from tensorflow import Tensor

from chessnn import MoveRecord, MOVES_MAP

tensorflow.compat.v1.disable_eager_execution()
assert regularizers


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NN(object):
    _model: models.Model

    def __init__(self, path=tempfile.gettempdir()) -> None:
        super().__init__()
        self._train_acc_threshold = 0.9
        self._validate_acc_threshold = 0.9

        self._model = self._get_nn()
        js = self._model.to_json(indent=True)
        cs = hashlib.md5(js.encode()).hexdigest()
        self._store_prefix = os.path.join(path, str(cs))

        fname = self._store_prefix + ".hdf5"
        if os.path.exists(fname):
            logging.info("Loading model from: %s", fname)
            self._model = models.load_model(fname)
        else:
            logging.info("Starting with clean model: %s", fname)
            with open(self._store_prefix + ".json", 'w') as fp:
                fp.write(js)

            with open(self._store_prefix + ".txt", 'w') as fp:
                self._model.summary(print_fn=lambda x: fp.write(x + "\n"))

            plot_model(self._model, to_file=self._store_prefix + ".png", show_shapes=True)

    def save(self):
        filename = self._store_prefix + ".hdf5"
        logging.info("Saving model to: %s", filename)
        self._model.save(filename, overwrite=True)

    def inference(self, data):
        inputs, outputs = self._data_to_training_set(data, True)
        res = self._model.predict_on_batch(inputs)
        out = [x for x in res[0]]
        return out

    def train(self, data, epochs, validation_data=None):
        logging.info("Preparing training set of %s...", len(data))
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


reg = regularizers.l2(0.01)
optimizer = "adam"  # sgd rmsprop adagrad adadelta adamax adam nadam


class NNChess(NN):
    def _get_nn(self):
        pos_shape = (8, 8, len(PIECE_TYPES) * 2)
        position = layers.Input(shape=pos_shape, name="position")
        pos_analyzed = position
        pos_analyzed = self.__nn_conv(pos_analyzed)
        # pos_analyzed = self.__nn_residual(pos_analyzed)
        # pos_analyzed = self.__nn_simple(pos_analyzed)

        # pos_analyzed = layers.concatenate([pos_analyzed2, pos_analyzed3])
        # pos_analyzed = layers.Dense(64, activation=activ_hidden, kernel_regularizer=reg)(pos_analyzed)

        out_moves = layers.Dense(len(MOVES_MAP), activation="softmax", name="eval")(pos_analyzed)

        model = models.Model(inputs=[position], outputs=[out_moves])
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def __nn_simple(self, layer):
        activ = "relu"  # linear relu elu sigmoid tanh softmax
        layer = layers.Flatten()(layer)
        layer = layers.Dense(len(MOVES_MAP) * 2, activation=activ, kernel_regularizer=reg)(layer)
        layer = layers.Dense(len(MOVES_MAP), activation=activ, kernel_regularizer=reg)(layer)
        return layer

    def __nn_residual(self, position):
        def relu_bn(inputs: Tensor) -> Tensor:
            bn = layers.BatchNormalization()(inputs)
            relu = layers.ReLU()(bn)
            return relu

        activ = "relu"  # linear relu elu sigmoid tanh softmax

        def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
            y = x
            # y = layers.Conv2D(kernel_size=(kernel_size, kernel_size), filters=filters, activation=activ, strides=1)(y)
            # y = relu_bn(y)
            # y = layers.Conv2D(kernel_size=(kernel_size, kernel_size), filters=filters, activation=activ)(y)

            # if downsample:
            #    x = layers.Conv2D(kernel_size=kernel_size, filters=filters, activation=activ)(x)
            y = layers.Flatten()(y)

            y = layers.Dense(filters, activation=activ, kernel_regularizer=reg)(y)
            y = relu_bn(y)
            y = layers.Dense(filters, activation=activ, kernel_regularizer=reg)(y)

            y = layers.Add()([y, x])
            y = relu_bn(y)

            return y

        t = position
        params = [
            (32, 7, True),
            (16, 5, True),
            (8, 3, True),
        ]
        for param in params:
            num_filters, ksize, downsample, = param
            t = residual_block(t, downsample=downsample, filters=num_filters, kernel_size=ksize)

        # t = layers.AveragePooling2D(4)(t)
        t = layers.Flatten()(t)

        return t

    def __nn_conv(self, position):
        activ = "relu"
        conv31 = layers.Conv2D(8, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(position)
        conv32 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv31)
        conv33 = layers.Conv2D(32, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv32)
        flat3 = layers.Flatten()(conv33)

        conv41 = layers.Conv2D(8, kernel_size=(4, 4), activation=activ, kernel_regularizer=reg)(position)
        conv42 = layers.Conv2D(16, kernel_size=(4, 4), activation=activ, kernel_regularizer=reg)(conv41)
        flat4 = layers.Flatten()(conv42)

        conv51 = layers.Conv2D(8, kernel_size=(5, 5), activation=activ, kernel_regularizer=reg)(position)
        conv52 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv51)
        flat5 = layers.Flatten()(conv52)

        conv61 = layers.Conv2D(8, kernel_size=(6, 6), activation=activ, kernel_regularizer=reg)(position)
        # conv62 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv61)
        flat6 = layers.Flatten()(conv61)

        conv71 = layers.Conv2D(8, kernel_size=(7, 7), activation=activ, kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat7 = layers.Flatten()(conv71)

        conv81 = layers.Conv2D(8, kernel_size=(8, 8), activation=activ, kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ, kernel_regularizer=reg)(conv71)
        flat8 = layers.Flatten()(conv81)

        conc = layers.concatenate([flat3, flat4, flat5, flat6, flat7, flat8])
        return conc

    def _data_to_training_set(self, data, is_inference=False):
        batch_len = len(data)

        inputs_pos = np.full((batch_len, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        out_evals = np.full((batch_len, len(MOVES_MAP)), 0.0)

        batch_n = 0
        for moverec in data:
            assert isinstance(moverec, MoveRecord)

            inputs_pos[batch_n] = moverec.position

            move = (moverec.from_square, moverec.to_square)
            if move != (0, 0):
                out_evals[batch_n][MOVES_MAP.index(move)] = moverec.eval

            batch_n += 1

        return [inputs_pos], [out_evals]

    def _moves_iter(self, scores):
        for idx, score in sorted(np.ndenumerate(scores), key=itemgetter(1), reverse=True):
            idx = idx[0]
            move = chess.Move(MOVES_MAP[idx][0], MOVES_MAP[idx][1])
            yield move
