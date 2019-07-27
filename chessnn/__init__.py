import collections
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
    comm_stack: list

    def __init__(self, comments: list):
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
            if self.comm_stack:
                log_rec = self.comm_stack.pop(0)
                if log_rec.ignore:
                    comm = "ign"
                else:
                    pass
                    comm = "%.2f" % (log_rec.get_eval())

                self.write_token(board.san(move) + " {%s} " % comm)
            else:
                self.write_token(board.san(move))

            self.force_movenumber = False


class BoardOptim(chess.Board):
    def __init__(self, fen=chess.STARTING_FEN, *, chess960=False):
        super().__init__(fen, chess960=chess960)
        self._fens = []
        self.comment_stack = []
        self.initial_fen = chess.STARTING_FEN

    def set_chess960_pos(self, sharnagl):
        super().set_chess960_pos(sharnagl)
        self.initial_fen = self.fen()

    def write_pgn(self, fname, roundd):
        journal = pgn.Game.from_board(self)
        journal.headers.clear()
        if self.chess960:
            journal.headers["Variant"] = "Chess960"
        journal.headers["FEN"] = self.initial_fen
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

    def can_claim_threefold_repetition2(self):
        """
        Draw by threefold repetition can be claimed if the position on the
        board occured for the third time or if such a repetition is reached
        with one of the possible legal moves.

        Note that checking this can be slow: In the worst case
        scenario every legal move has to be tested and the entire game has to
        be replayed because there is no incremental transposition table.
        """
        transposition_key = self._transposition_key()
        transpositions = collections.Counter()
        transpositions.update((transposition_key,))

        # Count positions.
        switchyard = []
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(),))

        while switchyard:
            self.push(switchyard.pop())

        # Threefold repetition occured.
        if transpositions[transposition_key] >= 3:
            return True

        return False

    def is_fivefold_repetition1(self):
        cnt = Counter(self._fens)
        return cnt[self._fens[-1]] >= 5

    def can_claim_draw1(self):
        return super().can_claim_draw() or self.fullmove_number > 100

    def push(self, move):
        super().push(move)
        self._fens.append(self.epd().replace(" w ", " . ").replace(" b ", " . "))

    def pop(self):
        self._fens.pop(-1)
        return super().pop()

    def get_position(self):
        pos = np.full((2, 8, 8, len(chess.PIECE_TYPES)), 0)
        for square in chess.SQUARES:
            piece = self.piece_at(square)

            if not piece:
                continue

            pos[int(piece.color)][square_file(square)][square_rank(square)][piece.piece_type - 1] = 1

        pos.flags.writeable = False

        return pos

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

    def _plot(board, matrix, position, fig, caption):
        """
        :type matrix: numpy.array
        :type position:  numpy.array
        :type fig: matplotlib.axes.Axes
        :type caption: str
        :return:
        """
        fig.axis('off')

        img = fig.matshow(matrix)

        for square in chess.SQUARES:
            f = square_file(square)
            r = square_rank(square)

            if any(position[int(chess.WHITE)][f][r]):
                color = chess.WHITE
            elif any(position[int(chess.BLACK)][f][r]):
                color = chess.BLACK
            else:
                continue

            piece_type = np.argmax(position[int(color)][f][r])
            piece_symbol = chess.PIECE_SYMBOLS[piece_type + 1]

            fig.text(r, f, chess.UNICODE_PIECE_SYMBOLS[piece_symbol],
                     color="white" if color == chess.WHITE else "black",
                     alpha=0.8, ha="center", va="center")

        fig.set_title(caption)

    def multiplot(board, memo, pos, wfrom, wto, pmoves):
        if not is_debug() or board.fullmove_number < 1:
            return

        pyplot.close("all")
        fig = pyplot.figure()
        fig, axes = pyplot.subplots(4, 2, figsize=(5, 10), gridspec_kw={'wspace': 0.01, 'hspace': 0.3}, )

        board._plot(wfrom, pos, axes[0][0], "wfrom")
        board._plot(wto, pos, axes[0][1], "wto")

        board._plot(pmoves, pos, axes[3][0], "possible moves")

        axes[3][1].axis("off")
        axes[3][1].set_title(memo + " - " + chess.COLOR_NAMES[board.turn] + "#%d" % board.fullmove_number)
        # pyplot.tight_layout()
        pyplot.show()
        logging.debug("drawn")


class MoveRecord(object):
    piece: chess.Piece

    def __init__(self, position, move, is_repeat, fifty_progress, piece=None) -> None:
        super().__init__()
        self.fifty_progress = fifty_progress
        self.is_repeat = is_repeat
        self.forced_eval = None
        self.ignore = False

        self.position = position
        self.piece = piece

        self.attacked = None
        self.defended = None
        self.threatened = None
        self.threats = None
        self.from_round = 0

        self.to_square = move.to_square
        self.from_square = move.from_square

    def __str__(self) -> str:
        return json.dumps({x: y for x, y in self.__dict__.items() if x not in ('forced_eval', 'kpis')})

    def __hash__(self):
        h = xxhash.xxh64()
        h.update(self.position)
        return sum([hash(x) for x in
                    (h.intdigest(), self.to_square, self.from_square, self.piece, self.fifty_progress, self.is_repeat)])

    def __eq__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        pself = xxhash.xxh64()
        pself.update(self.position)
        po = xxhash.xxh64()
        po.update(o.position)

        return pself.intdigest() == po.intdigest() and self.piece == o.piece and self.from_square == o.from_square \
               and self.to_square == o.to_square and self.is_repeat == o.is_repeat and self.fifty_progress == o.fifty_progress

    def __ne__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        raise ValueError()

    def get_eval(self):
        if self.forced_eval is not None:
            return self.forced_eval

        return 0.0

    def get_move_num(self):
        return 64 * self.from_square + self.to_square


def is_debug():
    return 'pydevd' in sys.modules
