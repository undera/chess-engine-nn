import logging
import sys

from chess import STARTING_FEN, Board, pgn

from player import Player


def record_results(brd, rnd):
    journal = pgn.Game.from_board(brd)
    journal.headers.clear()
    journal.headers["White"] = "Lisa"
    journal.headers["Black"] = "Karen"
    journal.headers["Round"] = rnd
    journal.headers["Result"] = brd.result(claim_draw=True)
    if brd.is_checkmate():
        journal.end().comment = "checkmate"
    elif brd.can_claim_fifty_moves():
        journal.end().comment = "50 moves claim"
    elif brd.can_claim_threefold_repetition():
        journal.end().comment = "threefold claim"
    elif brd.is_insufficient_material():
        journal.end().comment = "insufficient material"
    elif not any(brd.generate_legal_moves()):
        journal.end().comment = "stalemate"
    else:
        journal.end().comment = "by other reason"

    # exporter = pgn.StringExporter(headers=True, variations=True, comments=True)
    # logging.info("\n%s", journal.accept(exporter))
    logging.info("Game #%d: %s by %s, %d moves", rnd, journal.headers["Result"], journal.end().comment,
                 brd.fullmove_number)
    with open("last.pgn", "w") as out:
        exporter = pgn.FileExporter(out)
        journal.accept(exporter)


def play_one_game(pwhite, pblack, rnd):
    board = Board(STARTING_FEN)
    pwhite.board = board
    pblack.board = board

    while True:
        wmove = pwhite.get_move()
        board.push(wmove)
        if board.is_game_over(claim_draw=True) or not wmove:
            break

        bmove = pblack.get_move()
        board.push(bmove)
        if board.is_game_over(claim_draw=True) or not bmove:
            break
    record_results(board, rnd)
    return board


class PlayerCLI(Player):

    def _choose_best_move(self, halfmove_score):
        print("Opponent's move: %s" % self.board.move_stack[-1])
        while True:
            move_str = input("Enter next move: ")
            try:
                move = self.board.parse_san(move_str)
                break
            except ValueError as exc:
                logging.error("Wrong move, try again: %s", exc)

        return move


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    logging.basicConfig(level=logging.DEBUG)

    white = Player(0)
    black = Player(1)

    for rnd in range(1000):
        play_one_game(white, black, rnd)

        white.learn()
        black.learn()
