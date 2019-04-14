import chess
import numpy as np
from matplotlib import pyplot

from chessnn import MoveRecord, BoardOptim, nn


class Player(object):
    board: BoardOptim
    nn: nn.NN

    def __init__(self, color, net) -> None:
        super().__init__()
        self.color = color
        # noinspection PyTypeChecker
        self.board = None
        self.start_from = 0
        self.nn = net
        self.moves_log = []

    def makes_move(self):
        pos = self.board.get_position() if self.color == chess.WHITE else self.board.mirror().get_position()
        move, possible_moves = self._choose_best_move(pos)
        move_rec = self._mirror_move(move) if self.color == chess.BLACK else move

        before = self.board.get_evals(self.board.board_fen())

        self.board.push(move)

        self.board.turn = not self.board.turn
        after = self.board.get_evals(self.board.board_fen())
        self.board.turn = not self.board.turn

        balance = np.subtract(after, before)

        piece = self.board.piece_at(move.to_square)
        log_rec = MoveRecord(position=pos, move=move_rec, kpis=balance, piece=piece.piece_type,
                             possible_moves=possible_moves)

        # logging.debug("%d. %s %s", self.board.fullmove_number, move, log_rec["score"])
        self.moves_log.append(log_rec)
        self.board.comment_stack.append("%s %s" % (log_rec.get_score(), balance))

        not_over = move and not self.board.is_game_over(claim_draw=False)
        return not_over

    def _choose_best_move(self, pos):
        wfrom, wto, pos_moves = self.nn.query(pos[np.newaxis, ...])
        self.board.plot(pos_moves, "possible_predicted")

        if self.color == chess.BLACK:
            wfrom = np.fliplr(wfrom)
            wto = np.fliplr(wto)

            self.board.plot(wfrom, "wfrom-flipped")
            self.board.plot(wto, "wto-flipped")
        else:
            self.board.plot(wfrom, "wfrom")
            self.board.plot(wto, "wto")

        move_rating = self._gen_move_rating(wfrom, wto)

        possible_moves = np.full((8, 8), 0)
        for x in move_rating:
            possible_moves[chess.square_file(x[0].to_square)][chess.square_rank(x[0].to_square)] = 1
        assert possible_moves.any()

        self.board.plot(possible_moves, "possible moves")

        if self.board.fullmove_number <= 1 and self.board.turn == chess.WHITE:
            return move_rating[self.start_from][0], possible_moves

        move_rating.sort(key=lambda w: w[1] * w[2], reverse=True)

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
        return selected_move, possible_moves

    def _gen_move_rating(self, weights_from, weights_to):
        move_rating = []
        for move in self.board.generate_legal_moves():
            sw = weights_from[chess.square_file(move.from_square)][chess.square_rank(move.from_square)]
            dw = weights_to[chess.square_file(move.to_square)][chess.square_rank(move.from_square)]

            move_rating.append((move, sw, dw))
        return move_rating

    def get_moves(self):
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            result = 1
        elif result == '0-1':
            result = 0
        else:
            result = 0.5

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

        new_move = chess.Move(flip(move.from_square), flip(move.to_square),
                              move.promotion, move.drop)
        return new_move
