## Exploratory sharpness slices

This descriptive follow-up was added after inspecting Luna's completed results. It uses the existing frozen QC cutoff, without tuning a new threshold. The tables restrict the representative panel to reference-eligible photographs; all-case, challenge, development and aesthetic results remain in sharpness-slices.json. Low measured detail can reflect shallow focus, silhouettes or smooth subjects, so the lower band is not a verified blur label.

| Baseline | Lower-band n | Tech MAE | Tech bias | Other n | Tech MAE | Tech bias |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 69 | 0.490 | 0.371 | 207 | 0.298 | 0.055 |
| DeepSeek V4.1 Flash 1017 | 69 | 0.548 | 0.484 | 207 | 0.378 | 0.052 |
| Jev 1.14 | 69 | 0.451 | 0.321 | 206 | 0.321 | -0.027 |
| Liquid d1.1 | 69 | 0.331 | 0.222 | 207 | 0.388 | -0.187 |
| Q-ReAlign Lite | 69 | 0.445 | 0.404 | 207 | 0.494 | 0.388 |
| Luna + DeepSeek mean | 69 | 0.506 | 0.428 | 207 | 0.322 | 0.053 |

Paired technical MAE changes below are treatment minus baseline; negative values are improvements. These subgroup point estimates have no multiplicity-adjusted inference and do not replace the primary 300-image comparison.

| Model | Arm | Lower-band paired n | ΔMAE | Other paired n | ΔMAE |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | own_stats | 69 | -0.057 | 207 | 0.037 |
| GPT-6 Luna | swapped_stats | 69 | 0.103 | 207 | 0.096 |
| DeepSeek V4.1 Flash 1017 | own_stats | 69 | -0.078 | 207 | 0.032 |
| DeepSeek V4.1 Flash 1017 | swapped_stats | 69 | 0.129 | 207 | 0.176 |
| Luna + DeepSeek mean | own_stats | 69 | -0.069 | 207 | 0.043 |
| Luna + DeepSeek mean | swapped_stats | 69 | 0.119 | 207 | 0.142 |
