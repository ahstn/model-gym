# Prompts for synthetic visual ratings

Version: `visual-rating-v1`. Research checked on 2026-09-27. Status: proposed prompts, ready for a pilot; scoring accuracy has not been measured.

For each image, generate two distinct targets: **technical quality** and **aesthetics**. Store both on a 1–5 scale, where a larger value is better. Use separate requests in the pilot to reduce the risk that one judgment influences the other. Test the joint prompt as a cheaper alternative before using it at scale.

## 1. What the source projects establish

| Source | Finding checked | Design consequence |
|---|---|---|
| [Q-Align paper](https://proceedings.mlr.press/v235/wu24ah.html), [paper text](https://arxiv.org/html/2312.17090v1) | Human MOS labels are converted into five ordered words for training. Inference uses probabilities over those words. IQA and IAA are separate tasks. | Keep the task distinction and five anchors. A prompt on a general model does not reproduce the trained scorer. |
| [Q-Align scorer](https://github.com/Q-Future/Q-Align/blob/main/q_align/evaluate/scorer.py) | The quality and aesthetics scorers use separate short questions and answer stems. Both select five label logits. This repository implementation weights them from 0 to 1. | Record the score scale. Do not compare raw values from different wrappers without conversion. |
| [Q-ReAlign scorer](https://github.com/Q-Future/Q-ReAlign/blob/main/qalign/scorer.py) | Uses the same label-logit method, with prompts, answer stems, and weights supplied by configuration or records. | Keep synthetic data generation separate from the student's scoring interface. |
| [NIMA, Google Research](https://research.google/blog/introducing-nima-neural-image-assessment/) | Technical assessment concerns defects such as noise and blur. Aesthetics includes appeal and expression. NIMA learns distributions of human ratings. | Define two rubrics. A single teacher's output is not a human rating distribution. |
| [Q-Insight demo](https://github.com/bytedance/Q-Insight/blob/main/src/eval/demo_score.py), [paper](https://arxiv.org/html/2503.22679v2) | Its trained model emits a numeric quality rating from 1 to 5. Joint degradation training improves perception of defects; background blur can help the subject. | A numeric JSON score is a useful baseline. Judge defects in context rather than treating all softness as damage. |
| [Q-Instruct](https://github.com/Q-Future/Q-Instruct) | Teaches low-level visual description and question answering. | Request a short statement of visible evidence for audits. Do not require long generated explanations for every label. |
| [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score), [paper](https://arxiv.org/html/2501.11561) | Builds soft labels using human-score information and trains score distributions. | Preserve real rating spread when available. Treat elicited probabilities as a separate experiment. |
| [ArtiMuse](https://thunderbolt215.github.io/ArtiMuse-project/) | Uses expert attributes covering composition, visual elements, execution, creativity, communication, emotion, and overall assessment. | Use these as viewing criteria, without assuming universal attribute weights or inventing artistic context. |
| [MUSIQ, Google Research](https://research.google/blog/musiq-assessing-image-aesthetic-and-technical-quality-with-multi-scale-transformers/) | Resizing and cropping can change the content being assessed. Its examples also show that natural-image quality judgments can include aesthetic effects. | Preserve the full composition and define a viewing policy. Our strict technical target is a project choice, not a claim that existing MOS datasets isolate defects perfectly. |

The supplied [q-align-datasets repository](https://huggingface.co/datasets/q-future/q-align-datasets/tree/3da8831b98d683a024b1216720df0cc8a832e376) is a media bundle. Its file listing includes AVA, KonIQ, SPAQ, KADID, and several test sets. Its README is only 21 bytes and does not define a rating rubric. The prompts and label conversion must be understood from the paper and code. The mirror's MIT tag does not establish the rights of each source dataset; retain the [existing dataset terms review](../datasets/iqa.md).

### Definitions used in this prompt version

**Technical quality:** how clearly and cleanly the delivered image presents its content, considering visible defects and loss of important detail. Relevant factors include focus, noise, exposure, compression, and processing artifacts. Without a clean reference, this is perceived fidelity; it cannot establish exact fidelity to an original scene.

**Aesthetics:** the visual appeal and expressive success of the whole image. Relevant factors include composition, hierarchy, color, light, coherence, and impact. Aesthetics depends on the viewer, culture, and purpose. These prompts define a broad image-only rubric, not a universal measure of beauty or a verified model of AVA voters.

The two dimensions can interact without being interchangeable. A sharp but poorly arranged image can score high on technical quality and low on aesthetics. A grainy image with strong visual expression can show the reverse. A blank image may be technically clean; if blankness is the actual content, do not classify it as an input failure.

Prompt adherence, semantic plausibility, commercial usefulness, and video quality are outside these two targets. Do not silently add them to a combined quality score. Rendering defects can affect the two requested targets, but an impossible scene alone is not proof of a technical defect.

## 2. Ready-to-use prompts

| File | How to use it |
|---|---|
| [system.txt](../../prompts/visual-rating-v1/system.txt) | Shared system message for every request. |
| [technical-quality.txt](../../prompts/visual-rating-v1/technical-quality.txt) | Attach one image and use this as the user message for technical quality. |
| [aesthetics.txt](../../prompts/visual-rating-v1/aesthetics.txt) | Attach the same image in a fresh request and use this as the user message for aesthetics. |
| [joint.txt](../../prompts/visual-rating-v1/joint.txt) | Attach one image and request both ratings. Use only after comparing against separate requests. |
| [response.schema.json](../../prompts/visual-rating-v1/response.schema.json) | Local validation schema for either a single result or the joint response. Adapt schema features to each provider's supported subset. |

Start with separate calls and identical image bytes. Do not pass the first score, filename, source dataset, existing MOS, generation prompt, or another model's judgment into the second call. These can bias an image-only rating. If a generation prompt is needed later, create a separate prompt-adherence task.

The prompts request one numeric judgment and one short evidence sentence per task. The application merges separate responses under `technical_quality` and `aesthetics`. The following is an **illustrative format**, not a scored image:

```json
{
  "technical_quality": {
    "task": "technical_quality",
    "status": "ok",
    "score": 4.2,
    "evidence": "The main subject is clear, with slight noise visible in the dark background.",
    "limitations": []
  },
  "aesthetics": {
    "task": "aesthetics",
    "status": "ok",
    "score": 3.1,
    "evidence": "The colors work together, but clutter near the subject weakens the visual hierarchy.",
    "limitations": []
  }
}
```

The 0.1 step limits false precision. It does not establish interval-scale validity or calibrated agreement with humans. Validate the rubric against human-rated development data. Do not call these outputs MOS or Q-Align scores; call them `teacher_rating_1_5`.

### Why the primary prompt does not request probabilities

Q-Align reads model logits. NIMA and DeQA learn distributions from rating data. A general API model that writes five numbers into JSON provides neither of those measurements. The first pilot therefore uses simple scalar judgments. Retain a five-word-only baseline to test whether numeric output improves ranking or merely adds noise.

For the soft-label experiment, replace the scalar output instruction with: “Return five nonnegative weights in the order bad, poor, fair, good, excellent. The weights must sum to one and express your uncertainty over the rating levels.” Store this separately as `elicited_level_weights`; compute the weighted score in application code. Do not claim it describes a population of human raters. Compare it with repeated ratings and genuine label logits where the endpoint exposes all five at the same answer position. Missing top-k logits must not be filled with zero probabilities.

## 3. Scale and training conversion

For a scalar score `s` in [1,5], derive `score_0_1 = (s - 1) / 4`. If a display score is needed, `score_0_100 = 25 * (s - 1)`. These are arithmetic transforms, not fitted calibration or new measurements.

For complete label probabilities `p` in the order bad, poor, fair, good, excellent, compute `score_1_5 = sum(p[i] * (i + 1))`. Q-Align's repository scorer uses the equivalent 0–1 weights. Its paper describes the 1–5 form. Record the actual wrapper and weights when collecting reference predictions.

Q-Align training uses equal-width bins over each source score range. If we apply that convention to our fixed [1,5] teacher scale, use:

| Teacher score | Student level |
|---|---|
| [1.0, 1.8] | bad |
| (1.8, 2.6] | poor |
| (2.6, 3.4] | fair |
| (3.4, 4.2] | good |
| (4.2, 5.0] | excellent |

This is the proposed conversion for this synthetic source. It is not nearest-anchor rounding. Keep the raw teacher rating so later calibration does not require regeneration. Do not normalize each batch by its observed minimum and maximum. A batch of good images must remain good. Do not turn a single scalar into a claimed human distribution by assigning an arbitrary Gaussian variance.

Student supervision can retain Q-Align's short task question and the matching quality or aesthetics answer stem, followed by the selected level word. Keep the teacher's longer rubric and audit text outside the student's score-only target unless an explanation task is being tested explicitly.

## 4. Input and validation contract

The application owns the image ID, source hash, prompt version and hash, model and provider, request settings, preprocessing details, billed usage, and response timestamp. Do not ask the teacher to invent this metadata. Save the raw response beside the parsed label.

Use an orientation-corrected full image with preserved aspect ratio and no new lossy compression. Record both source dimensions and delivered dimensions. For the pilot, compare a full view at the planned size limit with a full view plus deterministic native crops. Mark crops explicitly. Composition is assessed from the full view; crops support detail inspection. Do not give aesthetics only a center crop.

Validate JSON, task names, score bounds, 0.1 increments, status, and allowed limitations. `unscorable` requires a null score; `ok` requires a numeric score. Do not change invalid output into a middle rating. Record API refusals, timeouts, and schema failures as pipeline failures rather than low-quality images. Use at most one format retry under a fixed policy, preserving both attempts. Repeated failure goes to review.

Limitations are routing hints, not probabilities or calibrated confidence. An `unsupported_domain` flag should trigger review before training. Severe visible degradation is still a valid low-quality example. A request is unscorable only when the image cannot actually be assessed.

## 5. Pilot checks before generating 250K labels

Use the [teacher selection plan](synthetic-label-teachers.md#5-pilot-and-promotion-decision) and split human calibration data from teacher-selection data. Keep final test sets untouched.

| Check | Evidence to inspect |
|---|---|
| Separate versus joint calls | Per-task human correlation, score drift, failures, and whether the two scores become needlessly similar. Use the same images and model settings. |
| Scalar versus five words | Ranking resolution, repeatability, and human agreement. More decimal values alone are not a gain. |
| Controlled defects | Add blur, noise, or compression to matched sources. Quality should usually fall with visible severity. Review floor effects and stylistic exceptions rather than forcing every pair to differ. |
| Intentional effects | Include portraits with soft backgrounds, grain, low-key lighting, abstract art, and monochrome images. Check for blanket penalties. |
| Technical/aesthetic separation | Include clean ordinary images and technically imperfect expressive images. Inspect human agreement within each group. |
| Content and praise bias | Compare score spread across subjects and media. Check unsupported praise and invented intent. Do not force equal score distributions. |
| Viewing policy | Check whether resizing removes defects or crops change aesthetic judgments. Lock preprocessing before scale-up. |
| Teacher disagreement | Obtain blind independent labels before comparing them. Use a second teacher on a stratified audit sample, with human review for persistent disagreements. |

Report raw PLCC, SRCC, sample counts, and source-level bootstrap intervals separately for technical quality and aesthetics. Inspect calibration and every domain regression. A plausible explanation is not evidence that the score is correct. Aesthetics requires its own promotion decision.

## 6. Cost change from the earlier estimate

The [earlier cost table](synthetic-label-teachers.md#4-cost-assumptions) assumed 300 text input tokens, 100 answer tokens, and one call per image. These fuller rubrics are longer. Two independent tasks also send the image twice. The old figure of about $649 for Gemini batch labels is not a quote for this two-task prompt pack.

For separate calls, add the measured costs of the technical and aesthetic requests. For the joint variant, count the image once but include both rubrics and both results. Use provider-reported input, answer, and reasoning tokens from the pilot; then recompute costs per valid image with both labels. No label-generation run or cost measurement was performed for this prompt version.
