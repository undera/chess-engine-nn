import logging
from abc import abstractmethod
from typing import List

import chess
import numpy as np
from chess.engine import SimpleEngine, INFO_SCORE

from chessnn import MoveRecord, BoardOptim, nn, is_debug, MOVES_MAP


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
        self.illegal_moves = []

    def get_moves(self, in_round):
        res = []
        for x in self.moves_log:
            x.in_round = in_round
            res.append(x)
        self.moves_log.clear()

        ill = []
        for x in self.illegal_moves:
            x.in_round = in_round
            ill.append(x)
        self.illegal_moves.clear()

        return res, ill

    def makes_move(self):
        move, geval, maps_predicted = self._choose_best_move()
        moverec = self._get_moverec(move, geval)
        if is_debug() and maps_predicted:
            maps_actual, maps_predicted = self._maps_for_plot(maps_predicted, moverec)
            plots = ["attacked", "defended", "possib_from", "possib_to", "move_from", "move_to"]
            self.board.multiplot("", plots, maps_predicted, maps_actual)
        self._log_move(moverec)
        self.board.push(move)
        if is_debug():
            logging.debug("%d. %r %.2f\n%s", self.board.fullmove_number, move.uci(), geval, self.board.unicode())
        not_over = move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)
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

        moverec.attacked, moverec.defended = bflip.get_attacked_defended()
        moverec.possible = bflip.get_possible_moves()

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
        self.invalid_moves = 0

    def _choose_best_move(self):
        if self.color == chess.WHITE:
            pos = self.board.get_position()
        else:
            pos = self.board.mirror().get_position()
        moverec = MoveRecord(pos, chess.Move.null(), None, self.board.fullmove_number, self.board.halfmove_clock)
        movegen, geval, maps = self.nn.inference([moverec])
        return self._scores_to_move(movegen), geval[0], maps

    def _scores_to_move(self, movegen):
        cnt = 0
        for move in movegen:
            if self.color == chess.BLACK:
                flipped = self._mirror_move(move)
                move = flipped

            if not self.board.is_legal(move):
                self.illegal_moves.append(self._get_moverec(move, 0.0))
                cnt += 1
                continue

            break
        else:
            logging.warning("No valid moves")
            move = chess.Move.null()
        logging.debug("Invalid moves skipped: %s", cnt)
        self.invalid_moves += cnt
        return move


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

        return result.move, forced_eval, None
