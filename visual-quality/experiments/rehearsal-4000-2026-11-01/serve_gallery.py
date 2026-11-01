"""Serve the frozen gallery and exact views on localhost without directory browsing."""

from __future__ import annotations

import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

DEFAULT = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01")


class GalleryHandler(BaseHTTPRequestHandler):
    root: Path

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        allowed_root = self.root.resolve()
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/gallery/index.html")
            self.end_headers()
            return
        if re.fullmatch(r"/originals/[0-9a-f]{64}\.png", path):
            target = self.root / "views" / path.removeprefix("/originals/")
        elif re.fullmatch(r"/eligibility-originals/[0-9a-f]{64}\.png", path):
            allowed_root = (self.root.parent / "eligibility-optimization-2026-11-01").resolve()
            target = allowed_root / "views" / path.removeprefix("/eligibility-originals/")
        elif path in ("/report.md", "/scores-long.csv"):
            target = self.root / "published" / path.lstrip("/")
        elif path.startswith("/gallery/"):
            relative = Path(path.removeprefix("/gallery/"))
            permitted = (
                (
                    relative.name
                    in {
                        "index.html",
                        "high-variance.html",
                        "style.css",
                        "data.js",
                        "eligibility-trial.html",
                        "eligibility-trial.js",
                        "eligibility-report.md",
                    }
                    and len(relative.parts) == 1
                )
                or (
                    len(relative.parts) == 1
                    and re.fullmatch(r"comparison-page-[1-4]\.(?:html|png)", relative.name)
                )
                or (
                    len(relative.parts) == 2
                    and relative.parts[0] == "previews"
                    and re.fullmatch(r"[CRSWU][0-9]{4}\.jpg", relative.name)
                )
            )
            if not permitted:
                self.send_error(404)
                return
            target = self.root / "published" / "gallery" / relative
        else:
            self.send_error(404)
            return
        if not target.resolve().is_relative_to(allowed_root) or not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        if target.name in {"data.js", "index.html"} or (
            target.suffix == ".html" and target.name.startswith("comparison-page-")
        ):
            content = content.replace(("file://" + str(self.root / "views") + "/").encode(), b"/originals/")
        if target.name == "index.html":
            trial_link = (
                b' &middot; <a href="eligibility-trial.html">Eligibility prompt comparison</a>'
                if (self.root / "published" / "gallery" / "eligibility-trial.html").is_file()
                else b""
            )
            content = content.replace(
                b"<nav>",
                b'<p><a href="high-variance.html">Explore 48 highest-disagreement images</a>'
                + trial_link
                + b"</p><nav>",
                1,
            )
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    GalleryHandler.root = args.data
    ThreadingHTTPServer(("127.0.0.1", args.port), GalleryHandler).serve_forever()
