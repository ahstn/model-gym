"""Offline C2PA checks with a fixed SDK policy and no producer-name disclosure."""

from pathlib import Path

PYTHON_SDK_VERSION = "0.38.0"
RUST_SDK_VERSION = "0.91.0"
MAX_INPUT_BYTES = 32 * 1024 * 1024
OFFLINE_SETTINGS = {
    "core": {"allowed_network_hosts": []},
    "verify": {
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
        "verify_after_reading": True,
        "verify_trust": True,
        "verify_timestamp_trust": True,
    },
}
# No trust, allowed_list or user_anchors override: retain the pinned SDK defaults.
TRUST_POLICY = "c2pa-python-0.38.0-rs-0.91.0-default-anchors-offline"
SDK_POLICY = {
    "python_sdk": PYTHON_SDK_VERSION,
    "rust_sdk": RUST_SDK_VERSION,
    "trust_policy": TRUST_POLICY,
    "trust_overrides": None,
    "settings": OFFLINE_SETTINGS,
}


def _summary(status: str, present=None, signature=None, binding=None, trust="unknown", claim=None) -> dict:
    return {
        "status": status,
        "manifest_present": present,
        "signature_valid": signature,
        "hard_binding_valid": binding,
        "trust_status": trust,
        "ai_generation_claim": claim,
    }


def _mime(header: bytes) -> str | None:
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _declared_origin(manifest: dict) -> bool | None:
    values = set()
    assertions = manifest.get("assertions", [])
    if not isinstance(assertions, list) or len(assertions) > 128:
        return None
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        if assertion.get("label") not in ("c2pa.actions", "c2pa.actions.v2"):
            continue
        data = assertion.get("data", {})
        actions = data.get("actions", []) if isinstance(data, dict) else []
        if not isinstance(actions, list) or len(actions) > 256:
            return None
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") != "c2pa.created":
                continue
            value = action.get("digitalSourceType")
            if value == "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia":
                values.add(True)  # noqa: FBT003 - set.add is not a configurable boolean argument
            elif value in (
                "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
                "http://cv.iptc.org/newscodes/digitalsourcetype/computationalCapture",
            ):
                values.add(False)  # noqa: FBT003 - set.add is not a configurable boolean argument
    return values.pop() if len(values) == 1 else None


def _validated_summary(reader) -> dict:
    manifest = reader.get_active_manifest()
    if not manifest:
        return _summary("no_manifest", present=False)
    active = (reader.get_validation_results() or {}).get("activeManifest", {})
    successes = {item["code"] for item in active.get("success", [])}
    failures = {item["code"] for item in active.get("failure", [])}
    signature = True if "claimSignature.validated" in successes else None
    signature_failures = {code for code in failures if code.startswith("claimSignature.")}
    binding_codes = {"assertion.dataHash.match", "assertion.boxesHash.match", "assertion.bmffHash.match"}
    binding_failures = {
        code
        for code in failures
        if code.startswith(("assertion.dataHash.", "assertion.boxesHash.", "assertion.bmffHash."))
    }
    binding = True if successes & binding_codes else None
    if signature_failures:
        signature = False
    if binding_failures:
        binding = False
    trust = "unknown"
    if "signingCredential.untrusted" in failures:
        trust = "untrusted"
    elif "signingCredential.trusted" in successes and reader.get_validation_state() == "Trusted":
        trust = "trusted"
    # Trust failure alone does not erase a valid signature or matching asset hash.
    invalid = bool(failures - {"signingCredential.untrusted"}) or reader.get_validation_state() == "Invalid"
    if invalid or signature is False or binding is False:
        status = "invalid"
    elif signature and binding:
        status = "valid"
    else:
        status = "unavailable"
    claim = _declared_origin(manifest) if status == "valid" else None
    return _summary(status, present=True, signature=signature, binding=binding, trust=trust, claim=claim)


def credential_state(summary: dict) -> str:
    """Keep trust, invalidity, missing credentials and unavailable checks separate."""
    status = summary["status"]
    if status == "no_manifest":
        return "missing"
    if status == "valid":
        if summary["signature_valid"] and summary["hard_binding_valid"] and summary["trust_status"] == "trusted":
            return "verified"
        return "untrusted"
    return status if status in ("invalid", "error", "unavailable") else "error"


AI_URIS = {
    f"http://cv.iptc.org/newscodes/digitalsourcetype/{name}"
    for name in ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")
}


def inspect_c2pa(path: str | Path) -> dict:
    """Validate once and retain declaration fields plus complete validation results."""
    details = {
        "declared_ai": False,
        "digital_source_types": [],
        "actions": [],
        "validation_results": {},
        "validation_state": None,
    }
    summary = _summary("unavailable")
    try:
        import c2pa
    except (ImportError, OSError):
        return {**summary, **details, "state": "unavailable", "read_error": "Pinned C2PA SDK is unavailable"}
    if c2pa.__version__ != PYTHON_SDK_VERSION or c2pa.sdk_version() != RUST_SDK_VERSION:
        return {
            **summary,
            **details,
            "state": "unavailable",
            "read_error": "C2PA SDK version differs from pinned policy",
        }
    source = Path(path)
    try:
        if source.stat().st_size > MAX_INPUT_BYTES:
            return {
                **_summary("error"),
                **details,
                "state": "error",
                "read_error": f"File exceeds pinned validator input limit of {MAX_INPUT_BYTES} bytes",
            }
        with source.open("rb") as stream:
            mime = _mime(stream.read(12))
            if mime is None:
                return {**_summary("error"), **details, "state": "error", "read_error": "Unsupported C2PA image format"}
            stream.seek(0)
            with (
                c2pa.Context.from_dict(OFFLINE_SETTINGS) as context,
                c2pa.Reader(mime, stream, context=context) as reader,
            ):
                summary = _validated_summary(reader)
                if summary["manifest_present"]:
                    manifest = reader.get_active_manifest() or {}
                    details["validation_state"] = reader.get_validation_state()
                    details["validation_results"] = reader.get_validation_results()
                    details["claim_generator_info"] = manifest.get("claim_generator_info", [])
                    for assertion in manifest.get("assertions", []):
                        if assertion.get("label") in ("c2pa.actions", "c2pa.actions.v2"):
                            details["actions"].extend(assertion.get("data", {}).get("actions", []))
                    details["digital_source_types"] = sorted(
                        {a["digitalSourceType"] for a in details["actions"] if a.get("digitalSourceType")}
                    )
                    details["declared_ai"] = bool(set(details["digital_source_types"]) & AI_URIS)
    except c2pa.C2paError.ManifestNotFound:
        summary = _summary("no_manifest", present=False)
    except c2pa.C2paError.RemoteManifest:
        summary = _summary("unavailable")
        details["read_error"] = "Remote manifests are disabled by the offline policy"
    except (OSError, ValueError, TypeError, KeyError, c2pa.C2paError) as error:
        summary = _summary("error")
        details["read_error"] = str(error)
    return {**summary, **details, "state": credential_state(summary)}
