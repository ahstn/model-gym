# OpenAI provenance check

**Partial: 15 of 160 images returned valid results.** One request failed with HTTP 502. The next returned HTTP 429 and asked for a 49-minute wait. The run stopped; 143 images remain unattempted. Read the [results and model comparison](partial-2026-10-12/report.md).

This track checks the existing held-out native images with the [Content Provenance API](https://developers.openai.com/api/docs/guides/content-provenance). It checks OpenAI-issued C2PA credentials and supported SynthID watermarks. A `not_detected` result leaves the image's origin unresolved; it is not a real-photo label.

The cohort has 80 real photos and 20 images from each of DALL-E 2, DALL-E 3, Firefly and Midjourney v5. These are the same image identities as the [Decider metadata trial](../metadata_v3/report.md). Decider saw 512-pixel JPEG crops, while this API receives unchanged native bytes. The comparison therefore measures two pipeline stages with different inputs.

The AI files come from the 2023 Synthbuster release. There are no current GPT-image positive controls, so this test cannot measure detection of current OpenAI exports. Luna and Gemini used other image cohorts; their earlier scores remain historical context.

## Results and evidence

The report retains detection outcomes, C2PA validation and issuer fields, latency, failures and per-image comparisons. It does not convert absent signals into general AI accuracy or assign a zero price when the response has no billing field.

Code, the frozen protocol and compact results can be tracked. Native images, the manifest, source capture and complete API records stay in ignored `data/openai-provenance-v1/`. The original metadata study is unchanged. Both studies pin image hashes; no labels, source names or prompts are sent in the multipart request.

## Run and verify

Run offline checks from `visual-triage` using its existing Python environment. API calls use `OPENAI_API_KEY` from the private shell environment. Each call submits one native file with its verified MIME type and a generic filename.

The original runner makes no retries and stops on any failure. Its protocol fixes the cohort, input bytes, request code and a minimum 6.2-second interval. Do not refreeze or overwrite the study.

One HTTP 502 response stopped the first pass. The separate [resume amendment](resume-amendment.json) allowed continuation after that failure and at most one retry per affected case, with a 176-attempt ceiling. The next request hit the rate limit before any retry took place. Both saved runners now block further calls. A further pass needs a reviewed rate-limit policy that respects the server's wait time. No background run is scheduled.

Original failures remain in `attempts.jsonl`; any permitted linked retries go to `retries.jsonl`. Rebuild the partial report into a fresh directory without API calls:

```sh
python tracks/openai_provenance/report.py --allow-partial --output-dir data/openai-provenance-v1/report-rebuild
python -m pytest tests/test_openai_provenance* -q
```

The amendment leaves the original protocol and request code unchanged. A successful retry can supply a case result, but the report still counts the failed attempt and its elapsed time. Request charges remain unreported unless a separate verified source establishes them.
