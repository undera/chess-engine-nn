import logging
from typing import List

import chess
import numpy as np
from chess.engine import SimpleEngine, INFO_SCORE

from chessnn import MoveRecord, BoardOptim, nn


class Player(object):
    moves_log: List[MoveRecord]
    board: BoardOptim
    nn: nn.NN

    def __init__(self, name, color, net) -> None:
        super().__init__()
        self.name = name
        self.color = color
        # noinspection PyTypeChecker
        self.board = None
        self.nn = net
        self.moves_log = []

    def makes_move(self, in_round):
        moverec: MoveRecord
        moverec, move, geval = self._choose_best_move()
        moverec.from_round = in_round

        self._log_move(moverec, move)

        self.board.push(move)

        logging.debug("%d. %r %.2f\n%s", self.board.fullmove_number, move.uci(), geval, self.board.unicode())

        not_over = move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)
        return not_over

    def _log_move(self, moverec, move):
        if move != chess.Move.null():
            self.moves_log.append(moverec)
            self.board.comment_stack.append(moverec)

    def _choose_best_move(self):
        pos = self.board.get_position() if self.color == chess.WHITE else self.board.mirror().get_position()

        moverec = MoveRecord(pos, chess.Move.null(), self.board.halfmove_clock / 100.0)
        scores4096, = self.nn.inference([moverec])  # , geval
        geval = [0]
        move = self._scores_to_move(scores4096)
        moverec.from_square = move.from_square
        moverec.to_square = move.to_square
        moverec.forced_eval = geval[0]
        return moverec, move, geval[0]

    def _scores_to_move(self, scores_restored):
        cnt = 0
        for idx, score in sorted(enumerate(scores_restored), key=lambda x: -x[1]):
            move = chess.Move(idx // 64, idx % 64)
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
        return move

    def get_moves(self):
        res = []
        for x in self.moves_log:
            res.append(x)
        self.moves_log.clear()
        return res

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


class Stockfish(Player):
    def __init__(self, color) -> None:
        super().__init__("Stockfish", color, None)
        self.engine = SimpleEngine.popen_uci("stockfish")

    def makes_move(self, in_round):
        result = self.engine.play(self.board, chess.engine.Limit(time=0.0100), info=INFO_SCORE)
        logging.debug("SF move: %s, %s, %s", result.move, result.draw_offered, result.info)

        pos = self.board.get_position() if self.color == chess.WHITE else self.board.mirror().get_position()

        record = MoveRecord(pos, result.move, self.board.halfmove_clock / 100.0)
        if result.info['score'].is_mate():
            record.forced_eval = 1
        elif not result.info['score'].relative.cp:
            record.forced_eval = 0
        else:
            record.forced_eval = -1 / abs(result.info['score'].relative.cp) + 1

        self._log_move(record, result.move, )

        self.board.push(result.move)

        not_over = result.move != chess.Move.null() and not self.board.is_game_over(claim_draw=False)
        return not_over
