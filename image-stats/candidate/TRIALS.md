# Trial record and decision · 5 November 2026

**Retain Smaug + explicit AI-declaration review flags + diagnostic image-stats.**
Keep Smaug's released 0.5 threshold. Do not adopt an ensemble or a threshold
fitted to this small dataset as a production operating point.

We first tested 11 generated mugs on the RTX 3090, then 50 real-source photos
and 50 scene-matched generations, ten each from Nano Banana Pro, GPT Image 2,
Riverflow 2.6 Pro, FLUX.2 Pro and Seedream 5 Pro. Four transformations per image
produced 500 correlated views, not 500 independent images. Selection/review
preceded detector scoring; no generated image was rejected by detector score.

| Detector | Native fakes caught /50 | Real photos flagged /50 | Consequence |
| --- | ---: | ---: | --- |
| Smaug | 24 | 0 | Best native ranking (AUC 0.9476); retain |
| OpenFake | 22 | 4 | No reliable advantage over Smaug |
| Thermostatic Low-Quality | 34 | 10 | Mug success hid substantial real false positives |
| OmniAID v2 | 7 | 2 | Weak recall on current generators |
| Sentry | 7 | 9 | JPEG/WebP lost all seven native detections |
| UnivFD | 0 | 0 | Missed every native fake at the tested threshold |
| Capcheck redirect | 38 | 40 | High recall accompanied by near-chance discrimination |
| Jev 1.14 | 16 | 1 | 80-candidate development search did not establish superiority |
| d1.1 | 10 | 0 | Low recall with its frozen prompt/run threshold |

All models fit on the 24 GB 3090. Smaug used approximately 0.94 GiB allocated
VRAM and 90.5 ms per predecoded image in the local batch-one comparison.

**Metadata changed the native result.** Explicit declarations identify 40/50
fakes and no selected real photos. Smaug OR declarations catches **46/50**
native fakes, still flagging **0/50** real photos. None of the credentials is
verified/trusted under the pinned offline policy: 34 are untrusted and six
invalid. These are review flags, not verified proof. Stripping removes this
benefit; it does not establish real origin or remove pixel watermarks.

At the retained threshold, stripped PNG/JPEG/raster copies each yield 24/50
fake detections and 0/50 real flags. WebP yields 29/50 and 3/50. Lowering Smaug
to 0.00314 catches 36/50 native fakes with zero observed native false positives,
but flags 7/50 real WebP copies; we did not adopt that fitted native threshold.

**Ensembles did not establish a robustness improvement.** Using the existing
40/10 scene-pair split, a conservative two-of-three vote catches 21/40 test
fake copies versus Smaug's 19/40, with zero real flags for both. Both catch only
3/10 fake originals in every form. A higher-recall log-odds ensemble ties
recalibrated Smaug at 31/40 copies, 6/10 fakes in every form and 1/10 real flags.
Default OR flags 5/10 real photos. Retain the simpler Smaug candidate.

**Future decisions:** validate on a larger, fresh, scene-separated set before
changing thresholds or adding models. Include realistic re-encoding and real
screenshots/recaptures; count copies as correlated. The existing set has been
examined repeatedly and cannot confirm production error rates. Image-stats
quality metrics remain separate from authenticity decisions. Training is not
approved. Generated images, exact prompts and final evidence records are
retained; retired exploratory code/results are locally archived outside Git.
