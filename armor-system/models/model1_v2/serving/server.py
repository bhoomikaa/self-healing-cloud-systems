"""Minimal HTTP server for Model 1 v2 (standard library only; SERVING.md section 3).

    .venv/bin/python -I armor-system/models/model1_v2/serving/server.py [--host 127.0.0.1] [--port 8081] [--checkpoint-root DIR]

GET  /healthz        200 {"status": "ok", ...} once checkpoints and artefacts are verified
GET  /v1/metadata    model version, checkpoint hashes, threshold, D1 node/channel layout, segment limits
POST /v1/predict     body model1.telemetry.v1 -> 200 model1.prediction.v1 | 400 {"error": {code, field, message}}
Errors: 400 invalid JSON or contract violation, 404 unknown path, 405 wrong method, 411 missing length, 413 body too large,
415 non-JSON content type, 500 unexpected failure (logged; no traceback in the response).
"""
import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ARMOR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, _p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, _p))

from model1_v2.serving import contract as ct  # noqa: E402

MAX_BODY = 64 * 1024 * 1024
log = logging.getLogger("model1.server")


def make_handler(service):
    lock = threading.Lock()                       # torch models are shared; requests are scored one at a time

    class Handler(BaseHTTPRequestHandler):
        server_version = "model1-v2"

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, err, field, message):
            self._send(code, {"error": {"code": err, "field": field, "message": message}})

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"status": "ok", "model": service.metadata()["model"]["name"], "version": service.metadata()["model"]["version"]})
            elif self.path == "/v1/metadata":
                self._send(200, service.metadata())
            elif self.path == "/v1/predict":
                self._error(405, "method_not_allowed", "$", "use POST")
            else:
                self._error(404, "not_found", "$", f"unknown path {self.path}")

        def do_POST(self):
            if self.path != "/v1/predict":
                return self._error(404, "not_found", "$", f"unknown path {self.path}")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._error(415, "unsupported_media_type", "Content-Type", "send application/json")
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                return self._error(411, "length_required", "Content-Length", "a numeric Content-Length is required")
            if int(length) > MAX_BODY:
                return self._error(413, "body_too_large", "$", f"body exceeds {MAX_BODY} bytes; send at most {ct.MAX_MINUTES} minutes per request")
            try:
                body = json.loads(self.rfile.read(int(length)))
            except (ValueError, UnicodeDecodeError) as e:
                return self._error(400, "invalid_json", "$", str(e))
            try:
                with lock:
                    out = service.handle(body)
            except ct.RequestError as e:
                return self._send(400, {"error": e.as_dict()})
            except Exception:                                  # noqa: BLE001  (never leak a traceback to the client)
                log.exception("prediction failed")
                return self._error(500, "internal_error", "$", "prediction failed; see the server log")
            self._send(200, out)

        def do_PUT(self):
            self._error(405, "method_not_allowed", "$", "use GET or POST")

        do_DELETE = do_PATCH = do_PUT

    return Handler


def serve(service, host="127.0.0.1", port=8081):
    return ThreadingHTTPServer((host, port), make_handler(service))


def main(argv=None):
    from model1_v2.serving.service import Model1Service
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("MODEL1_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MODEL1_PORT", "8081")))
    ap.add_argument("--checkpoint-root", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    svc = Model1Service.load(a.checkpoint_root)
    httpd = serve(svc, a.host, a.port)
    log.info("model1-v2 serving on http://%s:%d (checkpoints verified: %s)", a.host, a.port, ", ".join(s[:12] for s in svc.checkpoint_sha256))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
