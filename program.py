import logging

from chess import STARTING_FEN, Board, pgn

from objects import Player

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    board = Board(STARTING_FEN)

    white = Player(board, 0)
    black = Player(board, 1)

    while True:
        wmove = white.get_move()
        board.push(wmove)
        if board.is_game_over(claim_draw=True):
            break

        bmove = black.get_move()
        board.push(bmove)
        if board.is_game_over(claim_draw=True):
            break

    logging.info("Game ended: %s", board.result(claim_draw=True))
    with open("last.pgn", "w") as out:
        game = pgn.Game.from_board(board)
        game.headers.clear()
        game.headers["White"] = "Lisa"
        game.headers["Black"] = "Karen"
        game.end().comment = board.result(claim_draw=True)
        exporter = pgn.FileExporter(out)
        game.accept(exporter)
