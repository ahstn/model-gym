# Pinned Lite server

`lite_server.py` is the unchanged Q-ReAlign Lite GPU server from the historical trial. Its SHA256 remains `aaab73e53225093d30e1fbe800425c3b9754d7a5547013f735de8f358a408dbb`. Keep these executed source bytes intact when claiming compatibility with the recorded run; a server change requires new provenance.

`lite-server-identity.json` is the retained health identity used by the current Lite adapter. The [Lite runbook](../../experiments/labeling-10000-2026-11-01/lite-runbook.md) documents checkpoint, environment, authentication and accounting. Importing the server starts GPU model initialization, so deploy it explicitly in its model environment; it is not a normal package import.
