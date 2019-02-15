import logging

from objects import Board, STARTING_POSITION, Player

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    board = Board()
    board.from_fen(STARTING_POSITION)

    white = Player(board, 0)
    black = Player(board, 1)

    while True:
        wmove = white.get_move()
        board.make_move(wmove)
        if not board.is_playable():
            break

        bmove = black.get_move()
        board.make_move(bmove)
        if not board.is_playable():
            break
