import logging
import os
import time
import warnings
from abc import abstractmethod

# from https://github.com/tensorflow/tensorflow/issues/26691
# noinspection PyPackageRequirements
import absl.logging
import numpy as np
from chess import PIECE_TYPES

from chessnn import MoveRecord

# noinspection PyProtectedMember
logging.root.removeHandler(absl.logging._absl_handler)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
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
        return [x for x in res]

    def train(self, data, epochs, validation_data=None):
        logging.info("Preparing training set...")
        inputs, outputs = self._data_to_training_set(data, False)

        logging.info("Starting to learn...")
        cbpath = '/tmp/tensorboard/%d' % (time.time() if epochs > 1 else 0)
        cbs = [callbacks.TensorBoard(cbpath)]
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


reg = regularizers.l2(0.001)
activ_hidden = "sigmoid"  # linear relu elu sigmoid tanh softmax
activ_out = "softmax"  # linear relu elu sigmoid tanh softmax
optimizer = "nadam"  # sgd rmsprop adagrad adadelta adamax adam nadam


class NNChess(NN):
    def _get_nn(self):

        pos_shape = (8, 8, len(PIECE_TYPES) * 2)
        position = layers.Input(shape=pos_shape, name="position")
        # flags = layers.Input(shape=(1,), name="flags")
        main = self.__nn_simple(position)
        out_moves = layers.Dense(4096, activation=activ_out, name="moves")(main)
        # out_eval = layers.Dense(2, activation=activ_out, name="eval")(main)

        model = models.Model(inputs=[position], outputs=[out_moves])  # , flags # , out_eval
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      # loss_weights=[1.0, 0.1],
                      metrics=['categorical_accuracy'])
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

    def _nn_conv(self, position):
        conv1 = layers.Conv2D(16, kernel_size=(3, 3), activation=activ_hidden, kernel_regularizer=reg)(position)
        main1 = layers.concatenate([layers.Flatten()(conv1), layers.Flatten()(position)])
        main1 = layers.Dropout(0.25)(main1)
        dense1 = layers.Dense(128, activation=activ_hidden, kernel_regularizer=reg)(main1)

        conv2 = layers.Conv2D(32, kernel_size=(3, 3), activation=activ_hidden, kernel_regularizer=reg)(conv1)
        main2 = layers.concatenate([layers.Flatten()(conv2), dense1])
        main2 = layers.Dropout(0.1)(main2)
        dense2 = layers.Dense(128, activation=activ_hidden, kernel_regularizer=reg)(main2)

        conv3 = layers.Conv2D(64, kernel_size=(3, 3), activation=activ_hidden, kernel_regularizer=reg)(conv2)
        pool1 = layers.MaxPool2D()(conv3)
        main3 = layers.concatenate([layers.Flatten()(pool1), dense2])
        # main3 = layers.Dropout(0.1)(main3)

        # main3 = layers.Flatten()(main3)

        # main3 = layers.concatenate([main3, flags])
        # dense3 = layers.Dense(64, activation=activ_hidden, kernel_regularizer=reg)(main3)

        return main3

    def _data_to_training_set(self, data, is_inference=False):
        batch_len = len(data)

        inputs_pos = np.full((batch_len, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        inputs_flags = np.full((batch_len, 1), 0.0)
        out_moves = np.full((batch_len, 4096), 0.0)
        evals = np.full((batch_len, 2), 0.0)

        batch_n = 0
        for move_rec in data:
            assert isinstance(move_rec, MoveRecord)

            pos, evl, move = move_rec.position, move_rec.eval, move_rec.get_move_num()

            evals[batch_n][0] = evl if evl is not None else 0.5
            evals[batch_n][1] = 1.0 - evals[batch_n][0]
            inputs_pos[batch_n] = pos

            inputs_flags[batch_n][0] = move_rec.fifty_progress

            out_moves[batch_n][move] = 1.0

            batch_n += 1

        return [inputs_pos], [out_moves]  # , inputs_flags #  , evals
