# Evidence for the native-metadata trial

AIDE supports tests that combine different visual signals. It does not test EXIF, C2PA, Decider or prompt optimisation. This trial measures whether specific context helps Decider. The paper does not show that metadata prompting works.

## What the paper establishes

[A Sanity Check for AI-generated Image Detection, v3](https://arxiv.org/html/2406.19435v3) introduces the Chameleon set. It contains 14,863 real photos and 11,170 generated images. To keep a generated image, two annotators had to mistake it for a camera photo. This tests polished synthetic images that can defeat rules based on obvious defects.

| Paper result | Implication |
| --- | --- |
| AIDE's strongest Chameleon row: 65.77% accuracy, 26.80% AI recall and 95.06% real specificity | Accuracy hides missed AI images. Derived balanced accuracy is 60.93%; an all-real rule reaches 57.09% ordinary accuracy. |
| AIGCDetectBenchmark: full hybrid 92.77%, semantics alone 75.26%, pixel branches alone 76.70% | Combining learned features helped this detector. It does not predict a Decider gain. |
| Same benchmark after JPEG quality 50: 69.60%, down from 92.77% | Delivery transforms can remove useful forensic evidence. |

The [pinned implementation](https://github.com/shilinyan99/AIDE/tree/6725b710d5c437ab2f59792908ce0377dfc907de) uses DCT frequency scores to select 32×32 patches. It applies SRM residual filters and learned branches, then combines their output with OpenCLIP features. Scalar tone and texture values in a prompt do not reproduce AIDE.

Chameleon's classes still differ by source and category. The author repository requires academic use and an access request. The paper describes a signed agreement. Chameleon is a research reference here, not this trial's dataset. [Repository terms](https://github.com/shilinyan99/AIDE/tree/6725b710d5c437ab2f59792908ce0377dfc907de#-chameleon), [HF paper](https://huggingface.co/papers/2406.19435)

## Fixed trial design

Use 80 training, 80 development and 160 test images from VISION and Synthbuster. Each split has equal real and generated classes. Test Decider across seven arms, with identical delivered pixels and the same output schema:

| Arm | Context added to the image |
| --- | --- |
| `image_only` | None |
| `resolution` | Native and delivered dimensions and aspect ratios |
| `camera` | Native EXIF availability, ISO, exposure, f-number and focal length |
| `tone` | Encoded luma quantiles, tonal spread, endpoint fractions and RGB moments |
| `texture` | Adjacent luma differences, Laplacian variance and histogram entropy |
| `provenance` | C2PA discovery, validation, trust and declared-origin summary |
| `all_context` | All five context groups |

Measurements describe encoded pixels; they are not calibrated blur, noise or quality scores. Resolution and compression can expose dataset shortcuts, as shown in [Fake or JPEG?](https://arxiv.org/html/2403.17608). Source identifiers, camera names, software, GPS, timestamps, captions and generation prompts remain outside model context.

Each arm receives twenty GEPA proposal attempts under the same search rules. Each source group has eight images and receives mean `1 - Brier error` reward. Reflection uses the four individual **training** images with the lowest reward in the evaluated batch. Each image keeps its own arm's context. Full groups remain the evaluation unit. Only prompt context changes. Output fields and the AI threshold of 0.5 stay fixed.

Development scores select each arm's prompt. Keep the common baseline unless reward improves by at least 0.005. Freeze all seven selections before any test or control calls. Run the common baseline in every arm on the same test set. This isolates the effect of added context. Then compare each selected prompt with its arm's baseline. Unchanged prompts reuse baseline calls. Test outcomes do not select prompts, fields or thresholds.

## What metadata can and cannot establish

Native metadata coverage differs between VISION and Synthbuster. Equal class sizes do not give balanced provenance coverage. Results can reflect how each source was collected. There is no wholly unseen generator family. Phone model names differ across splits, but hardware families and scenes may overlap.

Two fixed rules test for simple metadata shortcuts on the development set. One predicts real if an EXIF container is present. The other predicts real if any numeric camera setting is present. Neither uses pixels or model calls. A metadata-only model arm is not run.

Extract EXIF and validate C2PA on preserved native bytes; hash the delivered image separately. EXIF is editable. A C2PA signature authenticates a signer's claim and its binding, not the truth of the scene. Keep absent, failed, unavailable and untrusted evidence distinct. An absent AI-generation claim does not establish camera origin. [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/ContentCredentials.html), [guidance](https://spec.c2pa.org/specifications/specifications/2.4/guidance/Guidance.html)

Controlled tests remove context or swap it with donor metadata. They keep the same pixels and labels. Report them separately from native accuracy. Donor metadata is marked as experimental. No camera values are invented or described as genuine recipient metadata. Run the validator again for C2PA byte controls. A copied manifest cannot keep a trusted-valid status without a content-binding check. Credential fixture tests check validation behaviour, not real-world detection accuracy.

## Results to report

Report accuracy, balanced accuracy, AI recall and real false-positive rate. Include Brier loss, AUROC and review rate. Use paired group-bootstrap intervals to compare arms. Break results down by source, generator, metadata availability and credential coverage. For controls, show changes in probabilities, labels and Brier loss separately.

Per-image records retain the arm, prompt and image hashes, label, probability, outcome, metadata state, cost and client latency. Report task and reflection spending separately. Preserve failed attempts. Include extraction and validation time when making an end-to-end speed claim. Bootstrap intervals do not remove source bias.

Checked on 12 October 2026. Three Exa searches requested 15 result slots; this is not a count of unique verified sources. Fetches are excluded. The HF paper connector was unavailable; its public paper API supplied metadata. Full paper text, pinned source files and retrieval hashes are in ignored `data/metadata-v3/research/`. This note records the design, not trial results.
