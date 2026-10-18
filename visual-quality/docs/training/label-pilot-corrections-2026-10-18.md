# Pilot score corrections: 18 October 2026

The review corrects **seven technical-quality scores** and **three evidence sentences** in the 12-image pilot. Two ratings remain unchanged. The corrected records are saved in `data/label-pilot-12/review-v1/reviewed-observations.jsonl`. The [versioned correction plan](evidence/label-pilot-corrections-2026-10-18.json) stores every reviewed value in Git, and the [export report](evidence/label-pilot-correction-report-2026-10-18.json) records the artifact hashes and decision counts.

These are assistant judgments pending human review. All training masks remain false. The source responses, teacher observations, prepared images, and source manifests remain unchanged. No additional provider calls were made.

## Review method

Two fresh assistant reviewers inspected separate groups of six frozen PNG views at their delivered resolution. Each received the technical-quality rubric but no teacher scores or explanations. The primary assistant inspected all 12 views and compared each rating with the rubric and both the original and fresh review evidence.

The review uses the existing anchors: **1 bad, 2 poor, 3 fair, 4 good, 5 excellent**. Scores remain in 0.1 steps. A plausible source score was retained when the difference was a preference between nearby values. Scores were corrected when the visible evidence supported a different anchor or contradicted the teacher's explanation. The review did not average judgments or fit a numeric adjustment.

The intervals in the plan express subjective uncertainty. They are not statistical confidence intervals, human MOS, or measured precision. A blank-looking but visible image can receive a low score; missing or uninspectable input must still receive null.

## Reviewed values

The order below matches the frozen view manifest. Full image IDs, view IDs, evidence, and review reasons are in the correction plan.

| Image | Original | Reviewed | Decision and visible basis |
|---|---:|---:|---|
| Stone building, Megalith `70e149d9…` | 4.2 | 4.2 | Retain. Architectural detail is clear; minor softness and noise have limited effect. |
| Card, VizWiz `00007963` | 2.6 | 2.6 | Retain. Blur and washed-out detail already support a poor-to-fair rating. |
| Noisy forest, distortion `542567c4…` | 2.6 | 2.6 | Correct evidence only. Dense noise is the main visible defect; tree outlines remain clear. |
| Sign and garden, Megalith `e6733c0f…` | 3.6 | 4.7 | Correct score and evidence. Sign text and foreground detail are sharp; selective background focus does not establish heavy compression damage. |
| Near-black frame, VizWiz `00022266` | 1.5 | 1.0 | Correct score and evidence. Almost no scene detail is visible, which supports the bad anchor. |
| Noisy building, distortion `4f061814…` | 3.6 | 3.1 | Correct score and evidence. Strong noise reduces fidelity across the image, while structural detail remains inspectable. |
| People on a boat, Megalith `3337c957…` | 3.2 | 3.8 | Correct score and evidence. Mild softness, fine noise, and local lamp clipping have limited effect on the main content. |
| Washed-out frame, VizWiz `00022541` | 1.5 | 1.0 | Correct score and evidence. Severe blur and clipping remove almost all useful detail. |
| Noisy people, distortion `481a54a7…` | 2.6 | 2.6 | Correct evidence only. Colored noise damages texture; faces and clothing remain interpretable. |
| Carved subject, Megalith `8254d303…` | 2.2 | 3.0 | Correct score and evidence. Noise, low contrast, and softness reduce detail, but central contours remain inspectable. |
| Obstructed fabric, VizWiz `00021283` | 3.2 | 2.1 | Correct score and evidence. A large blurred foreground obstruction hides most of the view. |
| Lightly noisy forest, distortion `7dd40d7b…` | 3.2 | 3.2 | Correct evidence only. Fine noise, bright highlights, and deep shadows reduce detail; heavy block damage is not established. |

The four synthetic examples all contain added Gaussian noise. This supports checking the noise descriptions, but it does not prove that their parent images contain no other defects. The prepared images remain the basis for each judgment. Source operator strength is not a substitute for a quality score.

## Artifact checks

The [review commands](label-pilot.md#apply-and-verify-reviewed-corrections) validate the original run before writing a separate export. Each reviewed record embeds its complete original observation, including the model, provider, request identity, usage, and original score. The review metadata identifies the assistant method and the exact plan hash. It does not attribute the corrected score to DeepSeek.

`verify-review` reconstructs the export from the pinned plan and source records. It rejects changed ratings, evidence, masks, provenance, or report fields. Repeating `apply-review` verifies and reuses the same output. A later revision requires a new plan and output directory.

| Check | Result |
|---|---|
| Reviewed records | 12; all retain source image and view identity |
| Score values | Finite 1–5 values in exact 0.1 steps |
| Decisions | 7 score-and-evidence corrections, 3 evidence-only corrections, 2 retained |
| Training status | `pending_human_review`; all task masks false; `training_ready=false` |
| Original observations SHA-256 | `153a1e5df5317ed890e49e0435d769f2eaac35480a66d6b8aacbfc5df6a950d4` |
| Reviewed observations SHA-256 | `3464f9a7c3f0fa2824d50779e99a61df2d92fe92e629620b226883dfd14206bc` |

## Effect on the next batch

The [earlier audit](label-pilot-validation-2026-10-18.md) still holds the conditional 100-image expansion. Correcting this cohort does not establish that the teacher will score new images well. Compare another approved teacher on the same views, add controlled blur and JPEG cases, and gather human anchor ratings before accepting labels for training. The full pilot's reported provider cost remains **$0.001243306**.
