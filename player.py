# Inspired by TCEC Season 14 - Superfinal
# where Leela was trying to fry Stockfish
import logging

import chess
import numpy

from nn import NN


class Player(object):
    board: chess.Board
    nn: NN

    def __init__(self, piece_index) -> None:
        super().__init__()
        self.piece_index = piece_index
        self.board = None
        self.nn = NN("%s.hdf5" % self.piece_index, piece_index)

    def makes_move(self):
        self.nn.after_their_move(self.board, -1 if self.piece_index else 1)
        move = self._choose_best_move()
        move.san = self.board.san(move)
        self.nn.record_for_learning(self.board, move)
        self.board.push(move)
        self.nn.after_our_move(self.board)
        not_over = move and not self.board.is_game_over(claim_draw=True)
        return not_over

    def _choose_best_move(self):
        halfmove_score = self.board.halfmove_clock / 100.0
        weights_from, weights_to = self.nn.query(self.board.board_fen(), halfmove_score, self.board.fullmove_number)
        move_rating = []
        for move in self.board.generate_legal_moves():
            sr = move.from_square // 8
            sf = move.from_square % 8
            sw = weights_from[sr][sf]

            dr = move.to_square // 8
            df = move.to_square % 8
            dw = weights_to[dr][df]

            score = abs(sw) * abs(dw) * numpy.sign(min(sw, dw))
            move_rating.append((move, score))
        move_rating.sort(key=lambda x: x[1], reverse=True)
        selected_move = move_rating[0][0] if move_rating else chess.Move.null()
        for move, score in move_rating:
            self.board.push(move)
            try:
                if self.board.can_claim_draw():
                    continue
            finally:
                self.board.pop()

            selected_move = move
            break
        return selected_move

    def learn(self):
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            score = -1 if self.piece_index else 1
        elif result == '0-1':
            score = 1 if self.piece_index else -1
        else:
            score = 0

        # logging.debug("Player #%s is learning...", self.piece_index)
        last_eval = self.nn.learn(score, -1 if self.piece_index else 1)
        self.nn.save("%s.hdf5" % self.piece_index)
        return last_eval
