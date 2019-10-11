import collections
import copy
import json
import logging
import sys
from collections import Counter

import chess
import numpy as np
from chess import pgn, SquareSet, SQUARES
from matplotlib import pyplot

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

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

    def write_pgn(self, wp, bp, fname, roundd):
        journal = pgn.Game.from_board(self)
        journal.headers.clear()
        if self.chess960:
            journal.headers["Variant"] = "Chess960"
        journal.headers["FEN"] = self.initial_fen
        journal.headers["White"] = wp.name
        journal.headers["Black"] = bp.name
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
        pos = np.full((8, 8, len(chess.PIECE_TYPES) * 2), 0)
        for square in chess.SQUARES:
            piece = self.piece_at(square)

            if not piece:
                continue

            int(piece.color)
            channel = piece.piece_type - 1
            if piece.color:
                channel += len(PIECE_VALUES)
            pos[chess.square_file(square)][chess.square_rank(square)][channel] = 1

        pos.flags.writeable = False

        return pos

    def get_attacked_defended(self):
        attacked = np.full(64, 0.0)
        defended = np.full(64, 0.0)

        our = self.occupied_co[self.turn]
        their = self.occupied_co[not self.turn]

        for square in SquareSet(our):
            for our_defended in SquareSet(self.attacks_mask(square)):
                defended[our_defended] = 1.0

        for square in SquareSet(their):
            for our_attacked in SquareSet(self.attacks_mask(square)):
                attacked[our_attacked] = 1.0

        return attacked, defended

    def _plot(self, matrix, position, fig, caption):
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
            f = chess.square_file(square)
            r = chess.square_rank(square)

            cell = position[f][r]
            if any(cell[:6]):
                color = chess.WHITE
            elif any(cell[6:]):
                color = chess.BLACK
            else:
                continue

            piece_type = np.argmax(position[int(color)][f][r])
            piece_symbol = chess.PIECE_SYMBOLS[piece_type + 1]

            fig.text(r, f, chess.UNICODE_PIECE_SYMBOLS[piece_symbol],
                     color="white" if color == chess.WHITE else "black",
                     alpha=0.8, ha="center", va="center")

        fig.set_title(caption)

    def multiplot(self, memo, predicted, actual):
        if not is_debug() or self.fullmove_number < 1:
            return
        pos = self.get_position()

        pyplot.close("all")
        # fig = pyplot.figure()
        fig, axes = pyplot.subplots(3, 2, figsize=(5, 10), gridspec_kw={'wspace': 0.01, 'hspace': 0.3})

        pmap = ["attacked", "defended"]
        for idx, param in enumerate(pmap):
            self._plot(np.reshape(predicted[idx], (8, 8)), pos, axes[idx][0], "pre " + param)
            self._plot(np.reshape(actual[idx], (8, 8)), pos, axes[idx][1], "act " + param)

        # axes[3][1].axis("off")
        # axes[3][1].set_title(memo + " - " + chess.COLOR_NAMES[self.turn] + "#%d" % self.fullmove_number)
        # pyplot.tight_layout()
        pyplot.show()
        logging.debug("drawn")

    def get_possible_moves(self):
        res = np.full(len(MOVES_MAP), 0.0)
        for move in self.generate_legal_moves():
            res[MOVES_MAP.index((move.from_square, move.to_square))] = 1.0
        return res


class MoveRecord(object):
    piece: chess.Piece

    def __init__(self, position, move, piece, move_number, fifty_progress) -> None:
        super().__init__()
        # TODO: add en passant square info
        # TODO: add castling rights info
        self.possible = None
        self.full_move = move_number
        self.fifty_progress = fifty_progress
        self.eval = None
        self.ignore = False

        self.position = position
        self.piece = piece

        self.from_round = 0

        self.to_square = move.to_square
        self.from_square = move.from_square

        self.attacked = None
        self.defended = None

    def __str__(self) -> str:
        return json.dumps({x: y for x, y in self.__dict__.items() if x not in ('forced_eval', 'kpis')})

    def get_eval(self):
        if self.eval is not None:
            return self.eval

        return 0.0

    def get_move_num(self):
        if self.from_square == self.to_square:
            return -1  # null move

        return MOVES_MAP.index((self.from_square, self.to_square))


def is_debug():
    return 'pydevd' in sys.modules


def _possible_moves():
    res = set()
    for f in SQUARES:
        for t in chess.SquareSet(chess.BB_RANK_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_FILE_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_DIAG_ATTACKS[f][0]):
            res.add((f, t))

        for t in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[f]):
            res.add((f, t))

    assert (10, 26) in res

    return list(sorted(res))


MOVES_MAP = _possible_moves()
