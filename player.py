# Inspired by TCEC Season 14 - Superfinal
# where Leela was trying to fry Stockfish

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
        self.nn = NN()

    def get_move(self):
        weights_from, weights_to = self.nn.query(self.board.board_fen(), self.board.halfmove_clock / 100.0)
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
        for move, score in move_rating:
            self.board.push(move)
            try:
                if self.board.can_claim_draw():
                    continue
            finally:
                self.board.pop()

            return move

        return move_rating[0][0] if move_rating else chess.Move.null()

    def learn(self):
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            score = -1 if self.piece_index else 1
        elif result == '0-1':
            score = 1 if self.piece_index else -1
        else:
            score = -0.25  # we play to win, not to draw

        batch = []
        playback = chess.Board(fen=self.board.starting_fen)
        for move in self.board.move_stack:
            batch.append((playback.board_fen(), move))
            playback.push(move)

        self.nn.learn(score, batch)
