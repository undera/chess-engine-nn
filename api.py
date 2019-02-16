import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from threading import Thread

from player import Player
from program import play_one_game


class ChessAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        #item = self.server.oqueue.get(False)
        item = "test"
        logging.debug("Sending move: %s", item)
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(bytes(str(item), 'ascii'))

    def do_POST(self):
        item = self.rfile
        logging.debug("Received move: %s", item)
        self.send_response(202)
        self.end_headers()
        self.wfile.write(bytes(str(item), 'ascii'))


class PlayerAPI(Player):

    def __init__(self, piece_index) -> None:
        super().__init__(piece_index)
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

    def _choose_best_move(self, halfmove_score):
        self.oqueue.put(self.board.move_stack[-1])
        logging.debug("Getting next move...")
        move_str = self.iqueue.get(True)
        return self.board.parse_san(move_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    white = Player(0)
    black = PlayerAPI(1)

    play_one_game(white, black, 1)

    # white.learn()
    # black.learn()
