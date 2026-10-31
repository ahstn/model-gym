# Backbone refresh research ledger

Research date: 2026-10-30. Previous verified backbone review: 2026-09-27, 33 days earlier. Initial broad discovery window: 2026-04-30 through 2026-10-30; new-release decisions use the period after the previous review. Older references and previously omitted candidates were retained separately.

Final analysis: [backbone-refresh-2026-10-30.md](../../backbone-refresh-2026-10-30.md).

## Six main rounds with each requested source

| Round | Exa work | Hugging Face work | Outcome |
|---|---|---|---|
| 1 | Recent 9–12B VLM releases and official launch discovery; 12 results requested | Search connector attempted, failed; official CLI retrieved 150 newest repositories matching size and image-text-to-text task; Qwen repository lookup checked | Established previous-review baseline and uncovered heavy derivative/quantization noise |
| 2 | October publisher/family coverage; 12 requested | Clef Flash, LensVLM and MiMo repository metadata | Separated real weight releases from updated cards |
| 3 | Visual benchmark/protocol audit; 10 requested | Qwen Base, Step3 Chat/Base, GLM Flash, Gemma IT | Verified total sizes and checkpoint types |
| 4 | Qwen MMMU-Pro baseline/protocol; 8 requested, plus original MMMU-Pro paper fetch | Taichu5, MiniCPM-o4.5, Mapika Decider and JEV | Checked newer scoring interfaces and borderline modality/parameter claims |
| 5 | Final release-gap search; 12 requested | Qwen Base, VLX-Seek, Granite Switch; initial Bonsai ID failed and was corrected later; expanded catalogue of 1,000 repositories | Broadened discovery beyond the initial list; found additional original task-specific checkpoints |
| 6 | PortraitAes primary-source validation; 5 requested; followed by ZwZ2 paper/card fetch | PortraitAes, FLUX prompt upsampler and ZwZ2 | Added direct portrait-aesthetics reference and the strongest new perception-focused starting checkpoint |

Additional HF lookup verified the correct Bonsai GGUF/MLX publisher repos and the community BF16 conversion. Failed searches or invalid repository identifiers were not treated as successful model discoveries.

## Accounting and deduplication

| Workstream | Exa search calls | Requested result slots |
|---|---:|---:|
| Main six-round review | 6 | 59 |
| Clef / LensVLM / MiMo / Mapika verification | 7 | 35 |
| Visual benchmark / trainability verification | 14 | 75 |
| Prior decisions / Granite / JEV / catalogue specialists | 6 | 30 |
| Total | **33** | **199** |

The Exa skill's `sources_reviewed` counter is defined as the sum of requested `numResults`. Accordingly, **199 is a result-slot count, including duplicates, not 199 unique validated sources**. The main stream returned 59 extracted entries, deduplicated to 47 exact URLs. Candidates appearing in several workstreams were merged into one assessment. Fetches and public Hub API/config/commit-history checks are additional and not included in that search counter.

Primary author cards, papers, release posts, code, licences and weight histories support the final conclusions. Discovery blogs and aggregators were used to find candidates, not as authority for benchmark or training claims. Self-reported benchmark results are not independent replications.

## Saved artifacts

- `search-audit.json`: the main round descriptions, deduplicable Exa URL/title/excerpt entries, HF connector outputs (including errors and the corrected Bonsai lookup), and workstream counters.
- `hf-round1-newest.json`: initial 150-model metadata catalogue.
- `hf-round5-expanded-catalogue.json`: 1,000-model metadata catalogue, from 2026-07-17 to 2026-10-30, including 409 repos created from 2026-09-28 onward. CLI serializes dates as `created_at`. `card_data` was empty for inspected records, so cards were fetched separately for licensing/training claims.

The catalogue filters use Hub parameter metadata, which can include misleading packed quantization counts. Original full-precision models and model configurations determine the final 9–12B eligibility decisions. New repository creation is a discovery signal, not proof of new original weights.

Official CLI query used for the expanded catalogue:

```sh
HF_HOME=/tmp/model-gym-hf-refresh-20261030 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  hf models list --num-parameters 'min:9B,max:12B' \
  --pipeline-tag image-text-to-text --sort created_at --limit 1000 \
  --expand createdAt,safetensors,cardData,library_name,pipeline_tag --format json
```

Direct checks also covered `any-to-any` MiniCPM-o and decision models tagged as text classification, which the catalogue task filter would miss. No model weights were downloaded and no inference or training jobs were run. A JEV adapter was counted from safetensors header metadata only.
