from collections import Counter
from random import shuffle

import chess
import numpy as np

from nn import NN, PIECE_MOBILITY


class Player(object):
    board: chess.Board
    nn: NN

    def __init__(self, color, nn) -> None:
        super().__init__()
        self.color = color
        self.board = None
        self.nn = nn
        self._moves_log = []

    def makes_move(self):
        move, fen = self._choose_best_move()
        move_rec = self._mirror_move(move) if self.color == chess.BLACK else move

        before = self._get_evals(fen)

        self.board.push(move)
        self.board.turn = not self.board.turn
        after = self._get_evals(fen)
        self.board.turn = not self.board.turn

        log_rec = {"move": move_rec, "fen": fen, "before": before, "after": after}
        #logging.debug("%d. %s %s %s", self.board.fullmove_number, move, log_rec["before"], log_rec["after"])
        self._moves_log.append(log_rec)

        not_over = move and not self.board.is_game_over(claim_draw=True)
        return not_over

    def _get_evals(self, fen):
        evals = [self._get_material_balance(fen), self._get_mobility(), self._get_attacks()]
        self.board.turn = not self.board.turn
        evals.append(self._get_attacks())
        self.board.turn = not self.board.turn
        return evals

    def _choose_best_move(self):
        fen = self.board.board_fen() if self.color == chess.WHITE else self.board.mirror().board_fen()
        weights_from, weights_to = self.nn.query(fen)
        if self.color == chess.BLACK:
            weights_from = np.flipud(weights_from)
            weights_to = np.flipud(weights_to)

        move_rating = self._gen_move_rating(weights_from, weights_to)
        move_rating.sort(key=lambda x: x[2], reverse=True)
        move_rating.sort(key=lambda x: x[1], reverse=True)

        selected_move = move_rating[0][0] if move_rating else chess.Move.null()
        if self.board.fullmove_number <= 1 and self.board.turn == chess.WHITE:
            shuffle(move_rating)
        for move, sw, dw in move_rating:
            self.board.push(move)
            try:
                if self.board.can_claim_draw():
                    continue
            finally:
                self.board.pop()

            selected_move = move
            break
        return selected_move, fen

    def _gen_move_rating(self, weights_from, weights_to):
        move_rating = []
        for move in self.board.generate_legal_moves():
            sr = move.from_square // 8
            sf = move.from_square % 8
            sw = weights_from[sr][sf]

            dr = move.to_square // 8
            df = move.to_square % 8
            dw = weights_to[dr][df]

            move_rating.append((move, sw, dw))
        return move_rating

    def get_moves(self):
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            score = 1
        elif result == '0-1':
            score = 0
        else:
            score = 0.5

        res = []
        for x in self._moves_log:
            x.update({"score": score})
            res.append(x)
        self._moves_log.clear()
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

        new_move = chess.Move(flip(move.from_square), flip(move.to_square),
                              move.promotion, move.drop)
        return new_move

    def _get_material_balance(self, fen):
        chars = Counter(fen)
        score = 0
        for piece in PIECE_MOBILITY:
            if piece in chars:
                score += PIECE_MOBILITY[piece] * chars[piece]

            if piece.lower() in chars:
                score -= PIECE_MOBILITY[piece] * chars[piece.lower()]

        return score

    def _get_mobility(self):
        moves = list(self.board.generate_legal_moves())
        mobility = len(moves)
        return mobility

    def _get_attacks(self):
        attacks = 0
        moves = list(self.board.generate_legal_moves())
        for move in moves:
            dest_piece = self.board.piece_at(move.to_square)
            if dest_piece:
                attacks += PIECE_MOBILITY[dest_piece.symbol().upper()]
        return attacks
