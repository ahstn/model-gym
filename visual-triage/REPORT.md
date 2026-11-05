# Initial visual-triage experiment

**Historical result, 2026-10-11:** this 30-image diagnostic supported Perplexity Decider as the next screening candidate. The [current benchmark report](BENCHMARKS.md) has the expanded comparison and recommendation. This run did not validate automatic AI-origin or NSFW rejection.

We made **159 paid API calls for $0.023713**, including provider checks and a complete Clef rerun. No images were removed, relabelled or used for training.

## Model comparison

Perplexity Decider offered the strongest quality signal at almost the same cost as Clef. The Gemini-to-Jev pipeline added cost and latency without a demonstrated quality benefit. Clef results below use the tested Cloudflare provider.

| Method | Valid outputs | Median / P95 latency | Cost for 30 attempts | Correct control medium | Known AI images flagged |
|---|---:|---:|---:|---:|---:|
| Perplexity Decider v1.1 27B | 30/30 | 0.473 / 0.658 s | $0.001663 | 6/6 | 3/6 |
| Clef Flash, Cloudflare | 30/30 | 0.894 / 1.690 s | $0.001587 | 6/6 | 2/6 |
| Gemini 2.5 Flash Lite | 29/30 | 1.608 / 2.775 s | $0.008197 | 6/6 | 1/6 |
| Gemini observations → Jev | 29/30 | 1.985 / 3.197 s | $0.009683 | 6/6 | 1/6 |

Latency includes local network overhead; requests ran in sequence with no automatic retries. One Gemini response failed validation because it marked an image unscorable but assigned an aesthetics score. Its cost remains included; the dependent Jev request was skipped.

## What the results support

Decider's technical-quality score decreased in eight of nine adjacent distortion steps. Clef decreased in seven, with two unavailable pairs; Gemini decreased in four and tied in five. These counts cover only three parent scenes.

Decider responded strongly to blur, scoring **3.63 → 3.49 → 2.42 → 1.62**. Its JPEG response was weaker: **4.19 → 4.22 → 4.16 → 3.93**. Compression needed further testing. Both direct decision models kept clean charts and renders at high technical quality. Those controls should not receive zero quality scores merely because they are not photographs.

AI-origin screening was unreliable. Two of Decider's three AI-suspected labels had probabilities below 0.5. Jev also assigned 0.93 camera-photo probability to one generated image after receiving a Gemini description. A text stage cannot recover visual evidence that the description omits.

There were **no confirmed explicit-content positives**. NSFW recall was not measured, so this run cannot support an automatic rejection gate. There were also no human quality or aesthetics ratings for this panel.

## Provider issue

The first Clef run mixed Cloudflare and Prime Intellect. It cost **$0.008766**, over five times the pinned run. Follow-up checks found different outputs for identical images: Cloudflare recognised solid red and blue, while Prime Intellect returned unknown. On three dataset probes, Cloudflare returned camera-photo and Prime Intellect returned diagram.

The [probe records](ARCHIVE.md) show a serving-path concern but do not establish its cause. Keep these results separate and pin the tested provider.

## Scope and evidence

The panel contained 12 original/distorted photos, six VizWiz images, six chart/render controls and six AGIQA-3K images. The AI examples shared a subject prompt; all 30 are now development material. Dataset provenance establishes the generated examples' origin. VizWiz recognisability labels are not quality ratings.

Each delivered image preserved its full frame at a maximum edge of 512 pixels, encoded as JPEG quality 92. Resizing can hide defects, so findings apply to that view. Source identities and labels were withheld from model requests.

See the [sample manifest](evidence/samples.json), [per-image results](ARCHIVE.md), [raw comparison](ARCHIVE.md), [pinned Clef run](ARCHIVE.md), [summary](ARCHIVE.md) and [research index](ARCHIVE.md) for detail. The original implementation passed 15 focused tests and lint checks; image hashes and evidence were checked for retained credentials and image request payloads.

Archived source snapshots and raw records require [archive restoration](ARCHIVE.md). The guide identifies the original paths and hashes.
