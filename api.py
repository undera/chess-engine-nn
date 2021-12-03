import logging
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
from queue import Queue
from threading import Thread

from chess import WHITE, BLACK

from chessnn.nn import NN, NNChess
from chessnn.player import NNPLayer
from training import play_one_game


class PlayerCLI(NNPLayer):

    def _choose_best_move(self):
        print("Opponent's move: %s" % self.board.move_stack[-1])
        while True:
            move_str = input("Enter next move: ")
            try:
                move = self.board.parse_san(move_str)
                break
            except ValueError as exc:
                logging.error("Wrong move, try again: %s", exc)

        return move


class ChessAPIHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path != '/move':
            return super().do_GET()

        logging.debug("Getting move to send...")
        item = self.server.oqueue.get(True)
        logging.debug("Sending move: %s", item)
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(str(item), 'ascii'))

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        item = self.rfile.read(content_len)
        item = item.decode('ascii')
        logging.debug("Received move: %s", item)
        self.server.iqueue.put(item)
        self.send_response(202)
        self.end_headers()
        self.wfile.write(bytes(str(item), 'ascii'))


class PlayerAPI(NNPLayer):

    def __init__(self, color) -> None:
        super().__init__("API", color, None)
        server_address = ('', 8090)
        self.httpd = HTTPServer(server_address, ChessAPIHandler)
        self.iqueue = Queue()
        self.oqueue = Queue()
        self.httpd.iqueue = self.iqueue
        self.httpd.oqueue = self.oqueue

        self.thr = Thread(target=self.run)
        self.thr.setDaemon(True)
        self.thr.start()

    def run(self):
        self.httpd.serve_forever()

    def _choose_best_move(self):
        if self.board.move_stack:
            self.oqueue.put(self.board.move_stack[-1])
        logging.debug("Getting next move from user...")
        move_str = self.iqueue.get(True)
        return self.board.parse_san(move_str), 0.5


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    white = PlayerAPI(WHITE)
    black = NNPLayer("Karen", BLACK, NNChess("nn.hdf5"))

    cnt = 1
    while True:
        rnd = random.randint(0, 960)
        play_one_game(white, black, 0)
        cnt += 1
