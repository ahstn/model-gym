import hashlib
import io

from PIL import Image, ImageCms, ImageOps, PngImagePlugin


def _render(payload: bytes, max_edge: int) -> tuple[bytes, dict]:
    with Image.open(io.BytesIO(payload)) as original:
        original.load()
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Fixed image views do not accept animated or multi-page inputs")
        native_size, native_mode = list(original.size), original.mode
        orientation = original.getexif().get(274, 1)
        icc = original.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(original)
        decoded_size = list(oriented.size)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        alpha = oriented.convert("RGBA").getchannel("A") if has_alpha else None
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        if icc is not None:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                cms_input = (
                    oriented if oriented.mode in ("RGB", "CMYK", "LAB", "L") else oriented.convert("RGB")
                )
                rgb = ImageCms.profileToProfile(cms_input, profile, srgb, outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError) as error:
                raise ValueError("Cannot convert embedded ICC profile to sRGB") from error
            colour_action = "embedded ICC converted to sRGB"
        else:
            rgb = oriented.convert("RGB")
            colour_action = "missing ICC assumed sRGB; Pillow mode conversion to RGB"
        if alpha is not None:
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha)
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb.info.clear()
        buffer = io.BytesIO()
        # A standard sRGB chunk avoids the creation timestamp in generated ICC profiles.
        png_info = PngImagePlugin.PngInfo()
        png_info.add(b"sRGB", b"\x00")
        rgb.save(buffer, format="PNG", pnginfo=png_info, optimize=False, compress_level=6)
        preprocessing = {
            "native_dimensions": native_size,
            "decoded_dimensions": decoded_size,
            "output_dimensions": list(rgb.size),
            "native_mode": native_mode,
            "exif_orientation": orientation,
            "orientation_applied": orientation not in (None, 1),
            "source_icc_sha256": hashlib.sha256(icc).hexdigest() if icc is not None else None,
            "colour_action": colour_action,
            "alpha_composited_on_white": has_alpha,
            "resized": list(rgb.size) != decoded_size,
            "crop_bounds": None,
        }
        return buffer.getvalue(), preprocessing


if __name__ == "__main__":
    import json
    import struct
    import sys
    import urllib.request
    import zlib
    from pathlib import Path

    request = json.load(sys.stdin)
    target = Path("/workspace/render-cache") / (request["sha256"] + ".png")
    target.parent.mkdir(exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == request["sha256"]:
        print(str(target))
        sys.exit(0)
    source = request["source"]
    url = source["url"]
    if not url.startswith("https://"):
        raise ValueError("Only public HTTPS source retrieval is permitted")
    headers = {}
    if source.get("kind") == "vizwiz":
        member = source["member"]
        start = member["offset"]
        end = min(source["archive_size"] - 1, start + member["compressed_size"] + 8192)
        headers = {"Range": f"bytes={start}-{end}", "If-Match": source["etag"]}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        payload = response.read(48 * 1024 * 1024 + 1)
        if len(payload) > 48 * 1024 * 1024:
            raise ValueError("Remote source exceeds declared byte limit")
        if source.get("kind") == "vizwiz":
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{source['archive_size']}"
            ):
                raise ValueError("VizWiz range response differs")
            filename_size, extra_size = struct.unpack_from("<HH", payload, 26)
            offset = 30 + filename_size + extra_size
            compressed = payload[offset : offset + member["compressed_size"]]
            payload = (
                zlib.decompress(compressed, -zlib.MAX_WBITS) if member["compression"] == 8 else compressed
            )
            if len(payload) != member["size"] or zlib.crc32(payload) != member["crc"]:
                raise ValueError("VizWiz member CRC differs")
    if hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError("Remote source bytes differ from frozen local acquisition")
    image, _ = _render(payload, source["target_edge"])
    if hashlib.sha256(image).hexdigest() != request["sha256"]:
        raise ValueError("Reconstructed delivered view differs from frozen local PNG")
    target.write_bytes(image)
    print(str(target))
