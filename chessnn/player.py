import logging
import random
from abc import abstractmethod
from typing import List

import chess
import numpy as np
from chess.engine import SimpleEngine, INFO_SCORE

from chessnn import MoveRecord, BoardOptim, nn, is_debug, MOVES_MAP, SYZYGY


class PlayerBase(object):
    moves_log: List[MoveRecord]
    board: BoardOptim

    def __init__(self, name, color) -> None:
        super().__init__()
        self.name = name
        self.color = color
        # noinspection PyTypeChecker
        self.board = None
        self.moves_log = []

    def get_moves(self, in_round):
        res = []
        for x in self.moves_log:
            x.in_round = in_round
            res.append(x)
        self.moves_log.clear()

        return res

    def makes_move(self):
        move, geval = self._choose_best_move()
        moverec = self._get_moverec(move, geval)
        self._log_move(moverec)
        self.board.push(move)
        if is_debug():
            logging.debug("%d. %r %.2f\n%s", self.board.fullmove_number, move.uci(), geval, self.board.unicode())
        not_over = move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)

        if len(self.board.piece_map()) <= 5:
            known = SYZYGY.get_wdl(self.board)
            not_over = False
            if known is not None:
                logging.debug("SyzygyDB: %s", known)
                if known > 0:
                    self.board.forced_result = chess.Outcome(chess.Termination.VARIANT_WIN, self.board.turn)
                elif known < 0:
                    self.board.forced_result = chess.Outcome(chess.Termination.VARIANT_LOSS, self.board.turn)
                else:
                    self.board.forced_result = chess.Outcome(chess.Termination.VARIANT_DRAW, self.board.turn)

        return not_over

    def _maps_for_plot(self, maps_predicted, moverec):
        maps_predicted = (maps_predicted[1], maps_predicted[2]) \
                         + self._decode_possible(maps_predicted[0]) + self._decode_possible(maps_predicted[3])
        mm = np.full(len(MOVES_MAP), 0.0)
        mm[moverec.get_move_num()] = 1.0
        maps_actual = (moverec.attacked, moverec.defended) \
                      + self._decode_possible(moverec.possible) + self._decode_possible(mm)
        maps_actual += self._decode_possible(maps_predicted[3])
        return maps_actual, maps_predicted

    def _get_moverec(self, move, geval):
        bflip: BoardOptim = self.board if self.color == chess.WHITE else self.board.mirror()
        pos = bflip.get_position()
        moveflip = move if self.color == chess.WHITE else self._mirror_move(move)
        piece = self.board.piece_at(move.from_square)
        piece_type = piece.piece_type if piece else None
        moverec = MoveRecord(pos, moveflip, piece_type, self.board.fullmove_number, self.board.halfmove_clock)
        moverec.eval = geval

        # moverec.attacked, moverec.defended = bflip.get_attacked_defended()
        # moverec.possible = bflip.get_possible_moves()

        return moverec

    def _flip64(self, array):
        a64 = np.reshape(array, (8, 8))
        a64flip = np.fliplr(a64)
        res = np.reshape(a64flip, (64,))
        return res

    def _log_move(self, moverec):
        if moverec.from_square != moverec.to_square:
            self.moves_log.append(moverec)
            self.board.comment_stack.append(moverec)

    def _mirror_move(self, move):
        """
        :type move: chess.Move
        """

        def flip(pos):
            arr = np.full((64,), False)
            arr[pos] = True
            arr = np.reshape(arr, (-1, 8))
            arr = np.flipud(arr)
            arr = arr.flatten()
            res = arr.argmax()
            return int(res)

        new_move = chess.Move(flip(move.from_square), flip(move.to_square), move.promotion, move.drop)
        return new_move

    @abstractmethod
    def _choose_best_move(self):
        pass

    def _decode_possible(self, possible):
        ffrom = np.full(64, 0.0)
        tto = np.full(64, 0.0)
        for idx, score in np.ndenumerate(possible):
            f, t = MOVES_MAP[idx[0]]
            ffrom[f] = max(ffrom[f], score)
            tto[t] = max(tto[t], score)

        return ffrom, tto


class NNPLayer(PlayerBase):
    nn: nn.NN

    def __init__(self, name, color, net) -> None:
        super().__init__(name, color)
        self.nn = net

    def _choose_best_move(self):
        if self.color == chess.WHITE:
            board = self.board
        else:
            board = self.board.mirror()

        pos = board.get_position()

        moves = []
        for move in board.generate_legal_moves():
            if not board.is_legal(move):
                logging.debug("Illegal: %s", move)
                continue

            moverec = MoveRecord(pos, move, None, board.fullmove_number, board.halfmove_clock)

            moverec.eval, = self.nn.inference([moverec])
            moves.append((moverec.eval, move))

        random.shuffle(moves)
        moves.sort(key=lambda x: x[0])
        geval, move = moves[-1]
        if self.color == chess.BLACK:
            move = self._mirror_move(move)

        return move, geval


class Stockfish(PlayerBase):
    def __init__(self, color) -> None:
        super().__init__("Stockfish", color)
        self.engine = SimpleEngine.popen_uci("stockfish")

    def _choose_best_move(self):
        result = self.engine.play(self.board, chess.engine.Limit(time=0.0100), info=INFO_SCORE)
        logging.debug("SF move: %s, %s, %s", result.move, result.draw_offered, result.info)

        if result.info['score'].is_mate():
            forced_eval = 1
        elif not result.info['score'].relative.cp:
            forced_eval = 0
        else:
            forced_eval = -1 / abs(result.info['score'].relative.cp) + 1

        return result.move, forced_eval
