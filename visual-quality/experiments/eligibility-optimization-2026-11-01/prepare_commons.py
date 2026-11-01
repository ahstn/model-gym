"""Download a fixed set of public-domain poster controls and preserve provenance."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")
TITLES = [
    "File:Exposition1874affiche.jpg",
    "File:1884 Worlds Industrial Cotton Exposition poster.jpg",
    'File:1st Woodcuts Exhibition of "Ryt" Association.jpg',
    "File:Advertisement for the Alaska-Yukon-Pacific Exposition, 1908 (AYP 1306).jpg",
    "File:Arnošt Hofbauer - Výstava spolku Mánes, Topičův salon (1898).JPG",
    "File:Arnošt Hofbauer XIII. Výstava Mánesa (1904).jpg",
    "File:Ausstellung der Theodor Graf'schen Sammlung Altgriechischer Porträts - Wien I.jpg",
    "File:Andreas Bloch - Den 3die Årlige Kunstudstilling 1886 - OK-17607 - National Museum of Art, Architecture and Design.jpg",
    "File:Canadian National Exhibition poster 1919.jpg",
    "File:Canadian War Memorials Exhibition, Royal Academy, Posters (CWM 19910001-479).jpg",
    "File:Ersatzmittel-Ausstellung... Wien 1918.jpg",
    "File:Expo sevilla 1929 poster.jpg",
    "File:Exposition d'ensemble du statuaire A. Rodin (Paris), 88 sculptures, 74 cadres-dessins. 10.V.-15.VII. 1902, Prague, Kinského Zahrada 9-7 H. protégé par la municipalité de Prague. affiche, Zupansky 1902.jpg",
    "File:Friends of the Fine Arts Society Jubilee Exhibition - Piotr Stachiewicz.jpg",
    "File:Gewerbe - Industrie - Ausstellung - Bodenbach 1897.jpg",
    "File:Expo-Milano-1906-Poster.jpg",
    "File:An Exhibition of Japanese Prints at the Grolier Club 2.jpg",
    "File:L'Exposition universelle de Paris, 1889.jpg",
    "File:Exhibition of posters - Federal Art Project Works Progress Administration LCCN98518243.jpg",
    "File:Berlin. Exposition industrielle, 1 mai-15 oct 1896.jpg",
]
UA = {"User-Agent": "ModelGymVisualQualityTrial/1.0 (public-domain research images)"}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as f:
                return f.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt * 2)


def item(pair):
    ix, page = pair
    cache = HERE / "review" / f"commons-{ix:04d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    i = page["imageinfo"][0]
    m = i.get("extmetadata", {})
    license = m.get("LicenseShortName", {}).get("value", "")
    if license not in ("Public domain", "CC0", "CC0 1.0"):
        raise ValueError("Unexpected license for " + page["title"] + ": " + license)
    url = i.get("thumburl") or i["url"]
    raw = get(url)
    original = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(original).convert("RGB")
    im.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    while True:
        b = io.BytesIO()
        im.save(b, format="PNG", optimize=True)
        view = b.getvalue()
        if len(view) <= 2650000:
            break
        im = im.resize((max(1, int(im.width * 0.9)), max(1, int(im.height * 0.9))), Image.Resampling.LANCZOS)
    sha = hashlib.sha256(view).hexdigest()
    p = DATA / "views" / f"{sha}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_bytes() != view:
        raise ValueError("View hash collision")
    p.write_bytes(view)
    record = {
        "case_id": f"W{ix:04d}",
        "family_id": "commons:" + page["title"],
        "source": "Wikimedia Commons",
        "source_title": page["title"],
        "source_page": i.get("descriptionurl"),
        "source_url": url,
        "source_license": license,
        "source_metadata": {
            k: v.get("value")
            for k, v in m.items()
            if k in ("LicenseShortName", "LicenseUrl", "Artist", "DateTimeOriginal", "Credit")
        },
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "image_path": str(p),
        "view_sha256": sha,
        "delivered_bytes": len(view),
        "width": im.width,
        "height": im.height,
    }
    cache.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    time.sleep(1)
    return record


def main():
    params = {
        "action": "query",
        "titles": "|".join(TITLES),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "1024",
        "format": "json",
    }
    pages = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))[
        "query"
    ]["pages"]
    by_title = {p["title"]: p for p in pages.values()}
    missing = [t for t in TITLES if t not in by_title or not by_title[t].get("imageinfo")]
    if missing:
        raise ValueError("Missing source metadata: " + repr(missing))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        cases = list(ex.map(item, enumerate([by_title[t] for t in TITLES], 1)))
    (HERE / "commons-sources.json").write_text(
        json.dumps(
            {
                "method": "Fixed selected Wikimedia Commons poster list, retrieved before benchmark assignments; both development and held-out include distinct works",
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"downloaded": len(cases), "bytes": sum(c["delivered_bytes"] for c in cases)}))


if __name__ == "__main__":
    main()
