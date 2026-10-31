# Continuation after the first strict validation failure

The original run stopped after A02. RunInfra returned HTTP 200, the requested model ID, and a known cost of $0.00030555. Its response wrote the score as `3.8999999999999999`. The existing validator parses decimals exactly and requires multiples of 0.1, so this is an invalid label. Preserve the original run and reply unchanged. Do not retry A02 or convert its output into an accepted observation.

The user requested the same 50 images. Continue with the remaining 40 requests that fit the provider's limit in a separate journal. Keep the original prompt, settings, image bytes, request hashes, and total $1 admission budget. The nine oversized requests remain excluded pending the user's choice. Each case may be attempted at most once across both journals.

This continuation changes one stopping rule after observing A02: a completed, priced reply that fails the strict rating contract can be recorded as invalid while later independent cases proceed. Stop for transport uncertainty, HTTP/provider errors, unexpected model identities, malformed envelopes, incomplete responses, or unknown costs. There are no retries or response repairs. Preserve all raw replies and keep invalid scores null in validated observation exports.

Compare validated outputs with the frozen assistant references using the original report. Any analysis of numeric values from rejected replies must be reported separately as a post-hoc diagnostic. It must preserve the actual returned values, identify the validation failures, and must not create accepted labels or change the full-50 decision rule.

Before reading new replies, six cases were selected for a small explanation audit using the old scores and frozen references: A18, A20, A28, A35, A40, and A46. These cover a clean/compressed controlled pair, two severe VizWiz failures, the largest old false-defect claim, and a sound old judgment. This targeted selection cannot estimate an explanation-support rate.

Observed visual criteria before reading their new replies:

| Case | Visible criteria |
| --- | --- |
| A18 | Clear fine branches and cloud structure, largely smooth sky; heavy blocks or banding lack support. |
| A20 | Large blocks and abrupt colour steps across most sky; lost cloud detail; branches remain identifiable. |
| A28 | Extreme blur, broad blue and pink shapes, almost no fine detail; no inspectable text. |
| A35 | Foreground obstruction covers nearly the whole frame; a small lower-left grass patch remains resolved. |
| A40 | Resolved faces, glasses, hands, clothing folds, wheelchair parts, and rails; heavy tiling or severe global blur lack support. |
| A46 | Defined stone face, carved hair, and surface texture, with mild softness toward the top and edges; natural texture is not severe image noise. |

All outputs remain pending human review. Aesthetics remains null and every task mask remains false. No reference, source image, historical observation, or original run record is changed.
