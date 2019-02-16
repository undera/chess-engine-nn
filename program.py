import logging

from chess import STARTING_FEN, Board, pgn

from player import Player

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    board = Board(STARTING_FEN)

    white = Player(board, 0)
    black = Player(board, 1)

    while True:
        wmove = white.get_move()
        board.push(wmove)
        if board.is_game_over(claim_draw=True) or not wmove:
            break

        bmove = black.get_move()
        board.push(bmove)
        if board.is_game_over(claim_draw=True) or not bmove:
            break

    # record results
    journal = pgn.Game.from_board(board)
    journal.headers.clear()
    journal.headers["White"] = "Lisa"
    journal.headers["Black"] = "Karen"
    journal.headers["Result"] = board.result(claim_draw=True)
    journal.end().comment = ""

    exporter = pgn.StringExporter(headers=True, variations=True, comments=True)
    logging.info("\n%s", journal.accept(exporter))

    with open("last.pgn", "w") as out:
        exporter = pgn.FileExporter(out)
        journal.accept(exporter)
