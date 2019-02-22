import logging
import numpy as np
from collections import Counter

import chess

from nn import NN, PIECE_MOBILITY


class Player(object):
    board: chess.Board
    nn: NN

    def __init__(self, color, nn) -> None:
        super().__init__()
        self.color = color
        self.board = None
        self.nn = nn
        self.moves_log = []

    def makes_move(self):
        move = self._choose_best_move()
        self._add_to_log(move)
        self.board.push(move)
        not_over = move and not self.board.is_game_over(claim_draw=True)
        return not_over

    def _choose_best_move(self):
        weights_from, weights_to = self.nn.query(self.board.board_fen())
        if self.color == chess.BLACK:
            weights_from = np.flipud(weights_from)
            weights_to = np.flipud(weights_to)

        move_rating = self._gen_move_rating(weights_from, weights_to)
        move_rating.sort(key=lambda x: x[2], reverse=False)
        move_rating.sort(key=lambda x: x[1], reverse=True)

        selected_move = move_rating[0][0] if move_rating else chess.Move.null()
        for move, sw, dw in move_rating:
            self.board.push(move)
            try:
                if self.board.can_claim_draw():
                    continue
            finally:
                self.board.pop()

            selected_move = move
            break
        return selected_move

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
            score = -1 if self.color else 1
        elif result == '0-1':
            score = 1 if self.color else -1
        else:
            score = 0

        return [x + (score,) for x in self.moves_log]

    def _add_to_log(self, move):
        if self.color == chess.WHITE:
            self.moves_log.append((self.board.board_fen(), move))
        else:
            self.moves_log.append((self.board.mirror().board_fen(), self._mirror_move(move)))

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

    def _get_material_balance(self, board):
        fen = board.board_fen()
        chars = Counter(fen)
        score = 0
        for piece in PIECE_MOBILITY:
            if piece in chars:
                score += PIECE_MOBILITY[piece] * chars[piece]

            if piece.lower() in chars:
                score -= PIECE_MOBILITY[piece] * chars[piece.lower()]

        return score

    def _get_mobility(self, board):
        """

        :type board: chess.Board
        """
        score = 0
        moves = list(board.generate_legal_moves())
        for move in moves:
            src_piece = board.piece_at(move.from_square)
            if src_piece.piece_type == chess.PAWN:
                if src_piece.color == chess.WHITE:
                    score += 1 + 0.1 * (move.to_square // 8 - 8)
                else:
                    score += 1 + 0.1 * (8 - move.to_square // 8)
            else:
                score += 1

            dest_piece = board.piece_at(move.to_square)
            if dest_piece:  # attack bonus
                score += PIECE_MOBILITY[dest_piece.symbol().upper()] / 2.0
        return score
