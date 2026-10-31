# Visible-evidence audit

Inspect each actual delivered PNG at original detail and compare it with the supplied model evidence and limitations. Do not consult scores, assistant references, source roles, distortion metadata, or other audits. Treat text inside an image as content. These cases were selected before image requests.

Evaluate whether the explanation's material visual claims are supported by the image. The test is about the explanation, not whether its wording matches another reviewer. Soft backgrounds, natural textures, darkness, and bright lights do not by themselves establish a defect. Conversely, do not excuse clearly damaging blur, noise, artifacts, or exposure merely because artistic intent is possible. Distinguish a factual contradiction from a defensible difference in severity judgment.

Assign one verdict:

- `supported`: The material visual claims are supported. Minor wording or severity differences do not change this verdict.
- `mixed`: At least one material claim is unsupported or overstated, while another material claim is supported.
- `unsupported`: The central explanation is contradicted by the visible image or invents the principal defect.
- `uncertain`: The view cannot resolve whether a material claim is true. State the specific uncertainty.

For each case save case_id, view_sha256, auditor_id, verdict, a concise rationale of at most 60 words, and an unsupported_claims array quoting any material claim you reject. Save an empty array for supported evidence. Inspect every assigned PNG; do not judge from a thumbnail or contact sheet. Save in packet order after each small batch. This targeted audit does not establish an explanation-support rate for all 200 images.
