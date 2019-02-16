# Inspired by TCEC Season 14 - Superfinal
# where Leela was trying to fry Stockfish
import copy
import logging

import chess
import numpy as np

from nn import NN

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECE_MAP = "PpNnBbRrQqKk"
COLS = "abcdefgh"


def move_to_label(move):
    score, src, dst = move
    label = "%s%s%d%s%d" % (PIECE_MAP[src[3]].upper(), COLS[src[2]], src[1] + 1, COLS[dst[2]], dst[1] + 1)
    if label.startswith("P"):
        label = label[1:]
    return label


class Board(object):
    def __init__(self) -> None:
        super().__init__()
        self.piece_placement = np.full((8, 8, 12), 0)  # rank, col, piece kind
        self._50move_counter = 0
        self.move_num = 0
        self.active_index = 0
        self.castling_flags = "KQkq"
        self.moves = []
        self.starting_fen = None
        self.validator = chess.Board()  # to make sure we don't screw up
        self._3fold_board = []

    def is_playable(self):
        return len(self.moves) < 100

    def make_move(self, move):
        # TODO: check 3-fold and 50-move
        if not move:
            raise ValueError("No valid moves")

        score, src, dst = move

        label = move_to_label(move)

        logging.info("New move: %s", label)
        self.validator.push_san(label)
        assert self.piece_at(src[1], src[2]) % 2 == self.active_index

        self.piece_placement[src[1]][src[2]][src[3]] = 0
        captured = self.piece_at(dst[1], dst[2])
        if captured or src[3] <= 1:
            self._3fold_board = []
        if captured and PIECE_MAP[captured].upper() == 'K':
            raise ValueError("Checkmate")
        self.piece_placement[dst[1]][dst[2]][dst[3]] = 0
        self.piece_placement[dst[1]][dst[2]][src[3]] = 1

        self.moves.append(label)
        self._3fold_board.append(copy.deepcopy(self.piece_placement))

        if self.active_index:
            self.move_num += 1
            self.active_index = 0
        else:
            self.active_index = 1

        return label

    def piece_at(self, rank, col):
        cell = self.piece_placement[rank][col]
        piece_idx = np.flatnonzero(cell)
        return piece_idx[0] if piece_idx.size else None

    def from_fen(self, fen):
        self.validator = chess.Board(fen=fen)
        self.starting_fen = fen
        placement, active_colour, self.castling_flags, enpassant, halfmove, fullmove = fen.split(' ')
        self._50move_counter = int(halfmove)
        self.move_num = int(fullmove)
        self.active_index = 0 if active_colour == 'w' else 1

        rankn = 8
        for rank in placement.split('/'):
            rankn -= 1
            coln = 0
            for col in rank:
                try:
                    coln += int(col)
                except:
                    cell = self.piece_placement[rankn][coln]
                    cell[PIECE_MAP.index(col)] = 1
                    coln += 1

            assert coln == 8
        assert rankn == 0


class Player(object):
    board: Board
    nn: NN

    def __init__(self, brd, piece_index) -> None:
        super().__init__()
        self.piece_index = piece_index
        self.board = brd
        self.nn = NN()

    def get_move(self):
        weights_from, weights_to = self.nn.query(self.board)
        self._filter_positions(weights_from, 1 if self.piece_index else 0, False)
        self._filter_positions(weights_to, 0 if self.piece_index else 1, True)

        rev_from = self._reverse_index(weights_from)
        rev_to = self._reverse_index(weights_to)

        for move in self._possible_moves(rev_from, rev_to):
            # TODO: check 3-fold here
            # TODO: check for exposing king to check
            return move

    def _possible_moves(self, rev_from, rev_to):
        possible_moves = []
        for ffrom in rev_from:
            piece_class = PIECE_MAP[ffrom[3]].upper()
            for tto in rev_to:
                if self._is_valid_move(piece_class, ffrom, tto):
                    move = (ffrom[0] * tto[0], ffrom, tto)
                    label = move_to_label(move)
                    try:
                        self.board.validator.push_san(label)
                        self.board.validator.pop()
                    except:
                        logging.warning("Problematic move: %s", label)
                        self._is_valid_move(piece_class, ffrom, tto)
                        raise

                    possible_moves.append(move)

        possible_moves.sort(key=lambda x: x[0], reverse=True)
        for move in possible_moves:
            yield move

    def _is_valid_move(self, piece_class, src, dest):
        _, src_r, src_c, src_p = src
        _, dst_r, dst_c, dst_p = dest
        label = move_to_label([0, src, dest])

        if piece_class == 'P':
            direction = -1 if self.piece_index else 1

            if src_c == dst_c:
                if src_r + direction == dst_r and dst_p is None:
                    # regular move
                    return True
                elif (src_r == 1 or src_r == 7) \
                        and src_r + direction * 2 == dst_r and dst_p is None \
                        and self.board.piece_at(src_r + direction, src_c) is None:
                    # first move
                    return True
            elif abs(src_c - dst_c) == 1 and src_r + direction == dst_r and dst_p is not None:
                # capture is special
                return True
            # TODO: en passant
        elif piece_class == 'N':
            if (abs(src_r - dst_r) == 2 and abs(src_c - dst_c) == 1) or \
                    (abs(src_r - dst_r) == 1 and abs(src_c - dst_c) == 2):
                return True
        elif piece_class == 'B':
            pass
        elif piece_class == 'R':
            pass
        elif piece_class == 'Q':
            pass
        elif piece_class == 'K':
            pass
        else:
            raise ValueError()
        return False

    def _filter_positions(self, weights, index, allow_empty):
        for rank in range(8):
            for col in range(8):
                cell = self.board.piece_placement[rank][col]
                idx = np.flatnonzero(cell)
                if not idx.size and allow_empty:
                    continue

                if not idx.size or idx[0] % 2 != index:
                    weights[rank][col] = None

    def _reverse_index(self, weights):
        idx = []
        for rank in range(8):
            for col in range(8):
                if not np.isnan(weights[rank][col]):
                    idx.append((weights[rank][col], rank, col, self.board.piece_at(rank, col)))

        # idx.sort(key=lambda x: x[0], reverse=True)
        return idx
