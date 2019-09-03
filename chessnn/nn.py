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

from chessnn import MoveRecord, MOVES_MAP

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
        return [x[0] for x in res]

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
        main = self.__nn_conv(position)

        # main = layers.Dense(64 * 2, activation="sigmoid", kernel_regularizer=reg)(main)

        out_attacked = layers.Dense(64, activation=activ_out, name="attacked")(main)
        out_defended = layers.Dense(64, activation=activ_out, name="defended")(main)

        conc = layers.concatenate([out_attacked, out_defended])
        main = layers.Dense(128, activation=activ_hidden)(main)
        main = layers.Dense(128, activation=activ_hidden)(main)
        out_moves = layers.Dense(len(MOVES_MAP), activation=activ_out, name="moves")(main)

        model = models.Model(inputs=[position], outputs=[out_moves, out_attacked, out_defended])
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      loss_weights=[1.0, 0.5, 0.5],
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

    def __nn_conv(self, position):
        conv31 = layers.Conv2D(8, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(position)
        conv32 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv31)
        conv33 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv32)
        flat3 = layers.Flatten()(conv33)

        conv51 = layers.Conv2D(8, kernel_size=(5, 5), activation="relu", kernel_regularizer=reg)(position)
        conv52 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv51)
        flat5 = layers.Flatten()(conv52)

        conv61 = layers.Conv2D(8, kernel_size=(6, 6), activation="relu", kernel_regularizer=reg)(position)
        # conv62 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv61)
        flat6 = layers.Flatten()(conv61)

        conv71 = layers.Conv2D(8, kernel_size=(7, 7), activation="relu", kernel_regularizer=reg)(position)
        # conv72 = layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_regularizer=reg)(conv71)
        flat7 = layers.Flatten()(conv71)

        conc = layers.concatenate([flat3, flat5, flat6, flat7])
        dense1 = layers.Dense(256, activation="sigmoid", kernel_regularizer=reg)(conc)

        # main2 = layers.concatenate([layers.Flatten()(conv2), flat1, dense0])
        # dense2 = layers.Dense(128, activation="sigmoid", kernel_regularizer=reg)(main2)

        # main3 = layers.concatenate([layers.Flatten()(conv3), dense2, flat1])
        # main3 = layers.Dropout(0.1)(main3)
        # dense3 = layers.Dense(256, activation="sigmoid", kernel_regularizer=reg)(main3)

        # concat = layers.concatenate([dense3])
        return dense1

    def _data_to_training_set(self, data, is_inference=False):
        batch_len = len(data)

        inputs_pos = np.full((batch_len, 8, 8, len(PIECE_TYPES) * 2), 0.0)
        inputs_flags = np.full((batch_len, 1), 0.0)
        out_moves = np.full((batch_len, len(MOVES_MAP)), 0.0)
        out_attacks = np.full((batch_len, 64), 0.0)
        out_threats = np.full((batch_len, 64), 0.0)
        evals = np.full((batch_len, 2), 0.0)

        batch_n = 0
        for moverec in data:
            assert isinstance(moverec, MoveRecord)

            pos, evl, move = moverec.position, moverec.eval, moverec.get_move_num()

            evals[batch_n][0] = evl
            evals[batch_n][1] = 1.0 - evals[batch_n][0]
            inputs_pos[batch_n] = pos

            inputs_flags[batch_n][0] = moverec.fifty_progress

            out_moves[batch_n][move] = evl
            na = np.count_nonzero(moverec.attacked)
            out_attacks[batch_n] = moverec.attacked / na if na else moverec.attacked
            nt = np.count_nonzero(moverec.defended)
            out_threats[batch_n] = moverec.defended / nt if nt else moverec.defended

            batch_n += 1

        return [inputs_pos], [out_moves, out_attacks, out_threats]  # , inputs_flags #  , evals
