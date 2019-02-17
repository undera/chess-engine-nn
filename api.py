import logging
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from queue import Queue
from threading import Thread

from player import Player
from program import play_one_game


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
        if self.board.move_stack:
            self.oqueue.put(self.board.move_stack[-1])
        logging.debug("Getting next move...")
        move_str = self.iqueue.get(True)
        return self.board.parse_san(move_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    white = PlayerAPI(0)
    black = Player(1)

    cnt = 1
    while True:
        play_one_game(white, black, cnt)

        white.learn()
        black.learn()
        cnt += 1
