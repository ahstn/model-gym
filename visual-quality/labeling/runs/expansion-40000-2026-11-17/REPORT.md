# 40K labeling aggregate report

The audited corpus contains 40,000 images, including 20,000 new images. The original 1,500 assistant references and first-1,000 Git publication remain unchanged. No new Q-ReAlign inference was run. Training masks remain disabled.

The ensemble remains 37.5% Luna, 37.5% DeepSeek and 25% Liquid for each outcome. Luna and DeepSeek use high reasoning. Image statistics remain separate review metadata. The new sample has no independent reference ratings; mean scores are not accuracy measures.

| Model | New numeric pairs | Mean technical | Mean aesthetic | Median final attempt (s) |
|---|---:|---:|---:|---:|
| luna | 20,000/20,000 | 3.56 | 3.40 | 15.19 |
| deepseek | 20,000/20,000 | 3.69 | 3.32 | 15.29 |
| liquid | 20,000/20,000 | 3.31 | 3.02 | 12.49 |
| jev | 19,998/20,000 | 3.48 | 3.00 | 12.11 |
| photo_ensemble | 20,000/20,000 | 3.55 | 3.27 | — |

Baseline outcomes and bounded recovery are shown separately. Recovery fees below are already included in the new-run API totals; compatibility probes are separate.

| Model | Baseline outcomes | Recovery cases / attempts | Recovery outcomes | Recovery known USD | Recovery unpriced attempts | Recovery exact USD |
|---|---|---:|---|---:|---:|---:|
| luna | valid: 20,000 | 0 / 0 | none | 0 | 0 | 0 |
| deepseek | failed: 5, valid: 19,995 | 5 / 5 | valid: 5 | 0.00738818 | 0 | 0.00738818 |
| liquid | valid: 20,000 | 0 / 0 | none | 0 | 0 | 0 |
| jev | failed: 2, valid: 19,998 | 0 / 0 | none | 0 | 0 | 0 |

Latency is measured from HTTP invocation through complete response receipt. Baseline grading ran four concurrent routes with up to 16 workers per route, so these are observed timings under shared load, not isolated provider benchmarks. Payload preparation and scheduler wait are excluded; retries retain separate timings and receipts.

EmbeddingGemma 2 grouped content into 64 clusters. There are 1,190 potential duplicate pairs for review; similarity does not establish duplicate identity or quality.

Full labels, observations, images, neighbors and duplicate identities remain in ignored local data. Only explicitly selected aggregate fields appear in this report and its companion summary.

Known costs include historical and new attempts without counting the derived ensemble twice. Unknown charges remain unknown; retries and probes are included. Reference and historical Lite prices are $0.00 by the user's reporting rule. Infrastructure is separate.

Embedding infrastructure: provider-posted billing checkpoint $6.1674128406448287; this is not a reconciled final total. The elapsed-runtime cost estimate is $6.168765284746023133333333333. Cleanup evidence is recorded.

| Model | New known API USD, including probes | Combined known API USD | Combined unpriced attempts | Exact combined API USD |
|---|---:|---:|---:|---:|
| luna | 16.306542325 | 32.526630525 | 167 | unknown |
| deepseek | 12.93037431 | 25.97683353 | 334 | unknown |
| liquid | 1.01343010 | 2.01913830 | 2898 | unknown |
| jev | 3.659049030 | 7.289058378 | 4882 | unknown |
| Reference | 0.00 | 0.00 | 0 | 0.00 |
| Q-ReAlign Lite (historical only) | 0.00 | 0.00 | 0 | 0.00 |
