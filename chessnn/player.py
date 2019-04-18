import logging

import chess
import numpy as np
import random

from chessnn import MoveRecord, BoardOptim, nn


class Player(object):
    board: BoardOptim
    nn: nn.NN

    def __init__(self, color, net) -> None:
        super().__init__()
        self.color = color
        # noinspection PyTypeChecker
        self.board = None
        self.start_from = (0, 0)
        self.nn = net
        self.moves_log = []

    def makes_move(self, in_round):
        info = self.board.get_info() if self.color == chess.WHITE else self.board.mirror().get_info()
        pos, attacked, defended, threatened, threats, material = info

        move, possible_moves, geval = self._choose_best_move(pos)

        self.board.push(move)

        self.board.turn = not self.board.turn
        info = self.board.get_info()
        apos, aattacked, adefended, athreatened, athreats, amaterial = info
        self.board.turn = not self.board.turn

        balance = [amaterial - material, aattacked.sum() - attacked.sum(), defended.sum() - adefended.sum(),
                   athreats.sum() - threats.sum(), athreatened.sum() - threatened.sum()]  # TODO: lost mobility

        piece = self.board.piece_at(move.to_square)
        move_rec = self._mirror_move(move) if self.color == chess.BLACK else move
        log_rec = MoveRecord(position=pos, move=move_rec, kpis=balance, piece=piece.piece_type,
                             possible_moves=possible_moves)
        log_rec.attacked = attacked
        log_rec.defended = defended
        log_rec.threatened = threatened
        log_rec.threats = threats
        log_rec.from_round = in_round

        logging.debug("%d. %r %.3f %.2f", self.board.fullmove_number, move, geval, log_rec.get_eval())
        self.moves_log.append(log_rec)
        self.board.comment_stack.append("%s %s" % (log_rec.get_eval(), balance))

        not_over = move and not self.board.is_game_over(claim_draw=False)

        afrom = np.full((8, 8), 0)
        afrom[chess.square_file(move_rec.from_square)][chess.square_rank(move_rec.from_square)] = 1
        ato = np.full((8, 8), 0)
        ato[chess.square_file(move_rec.to_square)][chess.square_rank(move_rec.to_square)] = 1
        self.board.multiplot("actual", pos, afrom, ato, attacked, defended, threats, threatened, possible_moves)

        return not_over

    def _choose_best_move(self, pos):
        geval, wfrom, wto, pmoves, attacks, defences, threats, threatened = self.nn.query(pos[np.newaxis, ...])  #

        self.board.multiplot("predicted", pos, wfrom, wto, attacks, defences, threats, threatened, pmoves)

        if self.color == chess.BLACK:
            wfrom = np.fliplr(wfrom)
            wto = np.fliplr(wto)

        move_rating, possible_moves = self._gen_move_rating(wfrom, wto)

        if self.board.fullmove_number <= self.start_from[1] + 1 and self.board.turn == chess.WHITE:
            logging.debug("Forcing %s move to be #%s", self.board.fullmove_number, self.start_from)
            if self.board.fullmove_number == 1:
                selected_move = move_rating[self.start_from[0]][0]
            else:
                selected_move = random.choice(move_rating)[0]
        else:
            move_rating.sort(key=lambda w: w[1] * w[2], reverse=True)
            selected_move = self._get_move_norepeat(move_rating)

        return selected_move, possible_moves, geval

    def _get_move_norepeat(self, move_rating):
        selected_move = move_rating[0][0] if move_rating else chess.Move.null()
        had_3fold = False
        for move, sw, dw in move_rating:
            self.board.push(move)
            try:
                if self.board.can_claim_threefold_repetition():
                    continue
            finally:
                self.board.pop()

            selected_move = move
            break

        if had_3fold:
            self.board.write_pgn("last.pgn", 0)
            logging.debug("Rolling back some moves from %s", len(self.moves_log))
            if len(self.moves_log) < 3:
                logging.warning("Less")

            self.moves_log.pop()
            self.moves_log.pop()
            self.moves_log.pop()

        return selected_move

    def _gen_move_rating(self, weights_from, weights_to):
        move_rating = []
        for move in self.board.generate_legal_moves():
            sw = weights_from[chess.square_file(move.from_square)][chess.square_rank(move.from_square)]
            dw = weights_to[chess.square_file(move.to_square)][chess.square_rank(move.from_square)]

            move_rating.append((move, sw, dw))

        possible_moves = np.full((8, 8), 0)
        for x in move_rating:
            possible_moves[chess.square_file(x[0].to_square)][chess.square_rank(x[0].to_square)] = 1
        assert possible_moves.any()

        if self.color == chess.BLACK:
            possible_moves = np.fliplr(possible_moves)

        return move_rating, possible_moves

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

        new_move = chess.Move(flip(move.from_square), flip(move.to_square),
                              move.promotion, move.drop)
        return new_move
