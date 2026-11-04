"""Start the experiment UI on your computer and open the browser."""
import threading
import webbrowser
from experiment_web import server

if __name__ == '__main__':
    try:
        httpd = server('127.0.0.1', 8080)
    except OSError:
        httpd = server('127.0.0.1', 0)
    url = f'http://127.0.0.1:{httpd.server_port}'
    print(f'Open {url} in your browser. Keep this terminal open; Ctrl+C stops the server.', flush=True)
    threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
