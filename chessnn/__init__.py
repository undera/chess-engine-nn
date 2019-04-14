import copy
import json
import logging
import sys
from collections import Counter

import chess
import numpy as np
import xxhash
from chess import pgn, square_file, square_rank
from matplotlib import pyplot

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.ROOK: 6,
    chess.QUEEN: 10,
    chess.KING: 100,
}


class MyStringExporter(pgn.StringExporter):

    def __init__(self, comments):
        super().__init__(headers=True, variations=True, comments=True)
        self.comm_stack = copy.copy(comments)

    def visit_move(self, board, move):
        if self.variations or not self.variation_depth:
            # Write the move number.
            if board.turn == chess.WHITE:
                self.write_token(str(board.fullmove_number) + ". ")
            elif self.force_movenumber:
                self.write_token(str(board.fullmove_number) + "... ")

            # Write the SAN.
            self.write_token(board.san(move) + " {%s} " % self.comm_stack.pop(0))

            self.force_movenumber = False


class BoardOptim(chess.Board):

    def __init__(self, fen=chess.STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self._fens = []
        self.comment_stack = []

    def write_pgn(self, fname, roundd):
        journal = pgn.Game.from_board(self)
        journal.headers.clear()
        journal.headers["White"] = "Lisa"
        journal.headers["Black"] = "Karen"
        journal.headers["Round"] = roundd
        journal.headers["Result"] = self.result(claim_draw=True)
        journal.headers["Site"] = self.explain()
        exporter = MyStringExporter(self.comment_stack)
        pgns = journal.accept(exporter)
        with open(fname, "w") as out:
            out.write(pgns)

    def explain(self):
        if self.is_checkmate():
            comm = "checkmate"
        elif self.can_claim_fifty_moves():
            comm = "50 moves"
        elif self.can_claim_threefold_repetition():
            comm = "threefold"
        elif self.is_insufficient_material():
            comm = "material"
        elif not any(self.generate_legal_moves()):
            comm = "stalemate"
        else:
            comm = "by other reason"
        return comm

    def can_claim_threefold_repetition1(self):
        # repetition = super().can_claim_threefold_repetition()
        # if repetition:
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 3

    def is_fivefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5

    def can_claim_draw1(self):
        return super().can_claim_draw() or self.fullmove_number > 100

    def push1(self, move):
        super().push(move)
        self._fens.append(self.epd().replace(" w ", " . ").replace(" b ", " . "))

    def pop1(self):
        self._fens.pop(-1)
        return super().pop()

    def get_info(self):
        pos = np.full((8, 8, 2, len(chess.PIECE_TYPES)), 0)
        attacked = np.full((8, 8,), 0)
        defended = np.full((8, 8,), 0)
        threatened = np.full((8, 8,), 0)
        threats = np.full((8, 8,), 0)
        material = 0
        for square in chess.SQUARES:
            piece = self.piece_at(square)

            if not piece:
                continue

            material += PIECE_VALUES[piece.piece_type] if piece.color == self.turn else -PIECE_VALUES[piece.piece_type]

            pos[square_file(square)][square_rank(square)][int(piece.color)][piece.piece_type - 1] = 1

            attackers = self.attackers(self.turn, square)
            attacked[square_file(square)][square_rank(square)] |= bool(attackers) and piece.color != self.turn

            defenders = self.attackers(self.turn, square)
            defended[square_file(square)][square_rank(square)] |= bool(defenders) and piece.color == self.turn

            threaters = self.attackers(not self.turn, square)
            threatened[square_file(square)][square_rank(square)] |= bool(threaters) and piece.color == self.turn
            if self.turn == piece.color:
                for tsq in threaters:
                    threats[square_file(tsq)][square_rank(tsq)] = 1

        pos.flags.writeable = False
        return pos, attacked, defended, threatened, threats, material

    def get_evals(self, fen):
        evals = [self._get_material_balance(fen), self._get_mobility(), self._get_attacks()]
        self.turn = not self.turn
        evals.append(self._get_attacks())
        self.turn = not self.turn
        return evals

    def _get_material_balance(self, fen):
        chars = Counter(fen)
        score = 0
        for piece in PIECE_VALUES:
            if piece in chars:
                score += PIECE_VALUES[piece] * chars[piece]

            if piece.lower() in chars:
                score -= PIECE_VALUES[piece] * chars[piece.lower()]

        if self.turn == chess.WHITE:
            return score
        else:
            return -score

    def _get_mobility(self):
        moves = list(self.generate_legal_moves())
        mobility = len(moves)
        return mobility

    def _get_attacks(self):
        attacks = 0
        moves = list(self.generate_legal_moves())
        for move in moves:
            dest_piece = self.piece_at(move.to_square)
            if dest_piece:
                attacks += PIECE_VALUES[dest_piece.symbol().upper()]
        return attacks

    def plot(self, matrix, position, caption):
        if not is_debug() or self.fullmove_number < 1:
            return

        img = pyplot.matshow(matrix)

        for square in chess.SQUARES:
            f = square_file(square)
            r = square_rank(square)
            piece = position[f][r]  # self.piece_at(square)

            if not piece.any():
                continue

            if piece[int(chess.WHITE)].any():
                color = chess.WHITE
                piece_type = np.argmax(piece[int(chess.WHITE)])
            else:
                color = chess.BLACK
                piece_type = np.argmax(piece[int(chess.BLACK)])

            piece_symbol = chess.PIECE_SYMBOLS[piece_type + 1]

            pyplot.text(r, f, chess.UNICODE_PIECE_SYMBOLS[piece_symbol],
                        color="white" if color == chess.WHITE else "black",
                        alpha=0.8, size="x-large", ha="center", va="center")

        pyplot.title(caption + " - " + chess.COLOR_NAMES[self.turn] + "#%d" % self.fullmove_number)
        pyplot.show()
        if is_debug() and self.fullmove_number >= 2:
            logging.debug("Stopping on move %s", self.fullmove_number)


class MoveRecord(object):
    piece: chess.Piece

    def __init__(self, position=None, move=None, kpis=None, piece=None, possible_moves=None) -> None:
        super().__init__()
        self.forced_score = None

        self.position = position
        self.piece = piece
        self.possible_moves = possible_moves

        self.attacked = None
        self.defended = None
        self.threatened = None
        self.threats = None
        self.from_round = 0

        self.to_square = move.to_square
        self.from_square = move.from_square
        self.kpis = [int(x) for x in kpis]

    def __str__(self) -> str:
        return json.dumps({x: y for x, y in self.__dict__.items() if x not in ('forced_score', 'kpis')})

    def __hash__(self):
        h = xxhash.xxh64()
        h.update(self.position)
        return sum([hash(x) for x in (h.intdigest(), self.to_square, self.from_square, self.piece)])

    def __eq__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        pself = xxhash.xxh64()
        pself.update(self.position)
        po = xxhash.xxh64()
        po.update(o.position)

        return pself.intdigest() == po.intdigest() and self.piece == o.piece and self.from_square == o.from_square and self.to_square == o.to_square

    def __ne__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        raise ValueError()

    def get_score(self):
        if self.forced_score is not None:
            return self.forced_score

        # material, attacks, defences, threats, threatened

        # first criteria
        if self.kpis[0] < 0:  # material loss
            return 0.0

        if self.kpis[4] > 0:  # threats up
            return 0.0

        if self.kpis[2] < 0:  # defence down
            return 0.0

        # second criteria
        if self.kpis[0] > 0:  # material up
            return 1.0

        if self.kpis[2] > 0:  # defence up
            return 1.0

        if self.kpis[4] < 0:  # threats down
            return 1.0

        # third criteria
        if self.kpis[1] > 0:  # attack more
            return 0.75

        if self.kpis[1] < 0:  # attack less
            return 0.0

        # fourth criteria
        # if self.kpis[1] > 0:  # mobility up
        #    return 0.5

        # if self.kpis[1] < 0:  # mobility down
        #    return 0.0

        # fifth criteria
        if self.piece == chess.PAWN:
            return 0.1

        return 0.0


def is_debug():
    return 'pydevd' in sys.modules
