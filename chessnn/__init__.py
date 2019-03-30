import copy
import json
from collections import Counter

import chess
from chess import pgn

PIECE_MOBILITY = {
    "P": 1,
    "N": 3,
    "B": 4,
    "R": 6,
    "Q": 10,
    "K": 100,
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


class MoveRecord(object):
    piece: chess.Piece

    def __init__(self, fen=None, move=None, kpis=None, piece=None) -> None:
        super().__init__()
        self.forced_score = None
        self.fen = fen
        self.piece = piece
        self.to_square = move.to_square
        self.from_square = move.from_square
        self.kpis = [int(x) for x in kpis]
        # TODO: add defences to KPIs

    def __str__(self) -> str:
        return json.dumps({x: y for x, y in self.__dict__.items() if x not in ('forced_score', 'kpis')})

    def __hash__(self):
        return sum([hash(x) for x in (self.fen, self.to_square, self.from_square, self.piece)])

    def __eq__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        return self.fen == o.fen and self.piece == o.piece and self.from_square == o.from_square and self.to_square == o.to_square

    def __ne__(self, o) -> bool:
        """
        :type o: MoveRecord
        """
        raise ValueError()

    def get_score(self):
        if self.forced_score is not None:
            return self.forced_score

        # first criteria
        if self.kpis[0] < 0:  # material loss
            return 0.0

        if self.kpis[3] > 0:  # threats up
            return 0.0

        # second criteria
        if self.kpis[0] > 0:  # material up
            return 1.0

        if self.kpis[3] < 0:  # threats down
            return 1.0

        # third criteria
        if self.kpis[2] > 0:  # attack more
            return 0.75

        if self.kpis[2] < 0:  # attack less
            return 0.0

        # fourth criteria
        if self.kpis[1] > 0:  # mobility up
            return 0.5

        if self.kpis[1] < 0:  # mobility down
            return 0.0

        # fifth criteria
        if self.piece == chess.PAWN:
            return 0.1

        return 0.0
