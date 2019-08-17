import logging
import os
import time
from abc import abstractmethod

# from https://github.com/tensorflow/tensorflow/issues/26691
# noinspection PyPackageRequirements
import absl.logging
import numpy as np
from chess import PIECE_TYPES

from chessnn import MoveRecord

# noinspection PyProtectedMember
logging.root.removeHandler(absl.logging._absl_handler)

from tensorflow.python.keras import models, layers, utils, callbacks, regularizers


class NN(object):
    _model: models.Model

    def __init__(self, filename=None) -> None:
        super().__init__()
        self._train_acc_threshold = 0.9
        self._validate_acc_threshold = 0.9
        if filename and os.path.exists(filename):
            logging.info("Loading model from: %s", filename)
            self._model = models.load_model(filename)
        else:
            logging.info("Starting with clean model")
            self._model = self._get_nn()
            self._model.summary(print_fn=logging.info)
            utils.plot_model(self._model, to_file=os.path.join(os.path.dirname(__file__), '..', 'model.png'),
                             show_shapes=True)

    def save(self, filename):
        logging.info("Saving model to: %s", filename)
        self._model.save(filename, overwrite=True)

    def inference(self, data):
        inputs, outputs = self._data_to_training_set(data, True)
        res = self._model.predict_on_batch(inputs)
        return [x[0] for x in res]

    def train(self, data, epochs, validation_data=None):
        logging.info("Preparing training set...")
        inputs, outputs = self._data_to_training_set(data, False)

        logging.info("Starting to learn...")
        cbs = [callbacks.TensorBoard('/tmp/tensorboard/%d' % time.time())] if epochs > 1 else []
        res = self._model.fit(inputs, outputs,  # sample_weight=np.array(sample_weights),
                              validation_split=0.1 if validation_data is None else 0.0, shuffle=True,
                              callbacks=cbs, verbose=2 if epochs > 1 else 0,
                              epochs=epochs, )
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


class NNChess(NN):
    def _get_nn(self):
        reg = regularizers.l2(0.001)
        activ_hidden = "relu"  # linear relu elu sigmoid tanh softmax
        activ_out = "softmax"  # linear relu elu sigmoid tanh softmax
        optimizer = "rmsprop"  # sgd rmsprop adagrad adadelta adamax adam nadam

        pos_shape = (8, 8, len(PIECE_TYPES) * 2)
        position = layers.Input(shape=pos_shape, name="position")
        flags = layers.Input(shape=(2,), name="flags")

        conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation=activ_hidden)(position)
        main1 = layers.concatenate([layers.Flatten()(conv1), flags])
        main1 = layers.Dropout(0.1)(main1)
        dense1 = layers.Dense(8, activation=activ_hidden)(main1)

        conv2 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ_hidden)(conv1)
        main2 = layers.concatenate([layers.Flatten()(conv2), dense1])
        main2 = layers.Dropout(0.1)(main2)
        dense2 = layers.Dense(16, activation=activ_hidden)(main2)

        conv3 = layers.Conv2D(8, kernel_size=(3, 3), activation=activ_hidden)(conv2)
        main3 = layers.concatenate([layers.Flatten()(conv3), dense2])
        main3 = layers.Dropout(0.1)(main3)
        dense3 = layers.Dense(32, activation=activ_hidden)(main3)

        main = layers.Dense(64)(dense3)

        out_moves = layers.Dense(4096, activation=activ_out, name="moves")(main)
        out_eval = layers.Dense(2, activation=activ_out, name="eval")(main)

        model = models.Model(inputs=[position, flags], outputs=[out_moves, out_eval])
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      loss_weights=[1.0, 0.5],
                      metrics=['categorical_accuracy'])
        return model

    def _data_to_training_set(self, data, is_inference=False):
        batch_len = len(data)

        inputs_pos = np.full((batch_len, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        inputs_flags = np.full((batch_len, 2), 0.0)
        out_moves = np.full((batch_len, 4096), 0.0)
        evals = np.full((batch_len, 2), 0.0)

        batch_n = 0
        for move_rec in data:
            assert isinstance(move_rec, MoveRecord)

            pos, evl, move = move_rec.position, move_rec.forced_eval, move_rec.get_move_num()

            evals[batch_n][0] = evl
            evals[batch_n][1] = 1.0 - evl
            inputs_pos[batch_n] = pos

            inputs_flags[batch_n][0] = move_rec.is_repeat
            inputs_flags[batch_n][1] = move_rec.fifty_progress

            out_moves[batch_n][move] = 1.0

            batch_n += 1

        return [inputs_pos, inputs_flags], [out_moves, evals]
