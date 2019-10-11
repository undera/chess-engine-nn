import logging
from abc import abstractmethod
from typing import List, Union

import chess
import numpy as np
from chess import BaseBoard
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

    def get_moves(self):
        res = []
        for x in self.moves_log:
            res.append(x)
        self.moves_log.clear()
        return res

    def makes_move(self, in_round):
        move, geval, maps_predicted = self._choose_best_move()
        moverec = self._get_moverec(move, geval, in_round)
        maps_actual = (moverec.possible, moverec.attacked, moverec.defended)
        if is_debug() and maps_predicted:
            self.board.multiplot("", maps_predicted[1:], maps_actual[1:])
        self._log_move(moverec)
        self.board.push(move)
        if is_debug():
            logging.debug("%d. %r %.2f\n%s", self.board.fullmove_number, move.uci(), geval, self.board.unicode())
        not_over = move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)
        return not_over

    def _get_moverec(self, move, geval, in_round):
        bflip: BoardOptim = self.board if self.color == chess.WHITE else self.board.mirror()
        pos = bflip.get_position()
        moveflip = move if self.color == chess.WHITE else self._mirror_move(move)
        piece = self.board.piece_at(move.from_square)
        piece_type = piece.piece_type if piece else None
        moverec = MoveRecord(pos, moveflip, piece_type, self.board.fullmove_number, self.board.halfmove_clock)
        moverec.from_round = in_round
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
        movegen, geval, possible, attacked, defended = self.nn.inference([moverec])
        return self._scores_to_move(movegen), geval[0], (possible, attacked, defended)

    def _scores_to_move(self, movegen):
        cnt = 0
        for move in movegen:
            if self.color == chess.BLACK:
                flipped = self._mirror_move(move)
                move = flipped

            if not self.board.is_legal(move):
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
