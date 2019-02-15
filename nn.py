import logging
import numpy as np

from keras import layers, Model
from keras.utils import plot_model


class NN(object):
    def __init__(self) -> None:
        super().__init__()
        self._model = self._get_nn()
        self._model.summary(print_fn=logging.debug)

    def _get_nn(self):
        positions = layers.Input(shape=(8 * 8 * 12,))  # 12 is len of PIECE_MAP
        hidden = layers.Dense(64, activation="sigmoid")(positions)
        hidden = layers.Dense(64, activation="sigmoid")(hidden)
        out_from = layers.Dense(64, activation="tanh")(hidden)
        out_to = layers.Dense(64, activation="tanh")(hidden)

        model = Model(inputs=[positions], outputs=[out_from, out_to])
        model.compile(optimizer='nadam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def query(self, brd):
        data = brd.piece_placement.flatten()[np.newaxis, ...]
        res = self._model.predict_on_batch(data)

        frm1 = res[0][0]
        frm2 = np.reshape(frm1, (-1, 8))
        tto1 = res[1][0]
        tto2 = np.reshape(tto1, (-1, 8))
        return frm2, tto2
