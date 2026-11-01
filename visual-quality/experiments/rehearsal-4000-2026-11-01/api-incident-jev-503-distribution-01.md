# Jev image-processing 503 distribution: read-only snapshot

The failures are dispersed across image sources, transformations, dimensions and payload sizes. These observations do not support a simple format or size filter. They are consistent with an intermittent service problem, but this one-attempt-per-image snapshot cannot establish the cause or rule out image-specific incompatibility.

The snapshot was created at **2026-11-01 15:25:43 UTC**, using observed responses received by **15:25:13 UTC**. It contains **1,945 distinct images**: **296 exact image-processing 503 responses (15.22%)** and **1,649 HTTP 200 responses**. The excluded request without an observed response is **R0600**, journaled at **14:08:51.583777 UTC** during the historical client interruption. It was not an in-flight request at snapshot time and has not been replayed. HTTP 200 is an availability outcome here, not a claim that every scoring response passed schema validation. No image score values were used in this diagnosis, and no API calls, retries or runtime changes were made.

| Group | Observed images | Exact 503s | Failure rate |
|---|---:|---:|---:|
| Benchmark | 500 | 78 | 15.60% |
| Rehearsal attempted so far | 1,445 | 218 | 15.09% |
| Megalith CC0 | 1,466 | 218 | 14.87% |
| VizWiz | 301 | 45 | 14.95% |
| Generated distortions | 127 | 19 | 14.96% |
| CLEVR | 26 | 3 | 11.54% |
| PlotQA | 25 | 11 | 44.00% |

Failures occur in all operator categories: blur 12/54, noise 5/61, JPEG compression 6/53, and no operator 273/1,777. Operator membership includes controlled-family parents where recorded in the frozen manifest; it should not be equated with the `generated_distortion` source count. Landscape and portrait failure rates are similar (15.48% and 15.07%). All benchmark partitions contain failures: development 12/100, representative 49/300, and challenge 17/100.

**No universal format or size boundary appears in this sample.** Every delivered image is a corpus-validated RGB PNG, including images whose original source file was JPEG. A supplementary read-only header check at **15:29:34 UTC** verified every one of the same 1,945 files against its frozen SHA256 and confirmed **PNG/RGB for all 296 failed and 1,649 HTTP 200 images**; there are no RGBA, palette or grayscale cases in this snapshot. This added verification did not include later requests or change the snapshot counts. Failed images range from 17,962 to 2,647,968 delivered bytes; HTTP 200 images range from 18,983 to 2,649,903 bytes. Median delivered sizes are 2,056,983.5 and 2,115,205 bytes respectively. Failed request bodies range from 25,750 to 3,532,422 bytes, while successful request bodies reach 3,535,002 bytes. All four size bins and all four pixel-count bins contain both outcomes. Forty-six exact width-by-height groups contain both outcomes, as do 17 related-image families. Related images are different byte sequences, so these comparisons are not successful retries of the same image.

PlotQA's 44% failure rate is an association worth retaining in the incident record. Its sample has only 25 attempts, and request timing was not randomized against provider conditions. The elevated PlotQA rate and differences between transformation categories therefore do not justify excluding those images or changing their encoding. Likewise, overlapping payload sizes do not rule out more complex content-dependent failures.

The exact error class was matched with `api_recovery.known_image_failure`: Jev, failed status, HTTP 503, no reported price, a verified raw-response hash, no transport exception, and the exact nested error body reporting **“Image processing is unavailable.”** That message does not identify an unsupported format or a size limit. The machine-readable [snapshot](api-incident-jev-503-distribution-01.json) preserves each observed attempt's metadata, receipt hashes, grouped counts and numerical distributions; it contains no technical or aesthetic labels.

Keep the frozen image inputs and missing labels intact. Combine this distribution check with the separate consecutive/rolling failure and cost audit when deciding whether to continue or stop the route. A later, explicitly documented retry comparison of the same failed images and matched controls could distinguish persistent image-specific failures from transient availability; this diagnostic performed no such retry.
