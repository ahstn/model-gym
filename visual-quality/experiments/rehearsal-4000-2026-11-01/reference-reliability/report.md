# Blind repeat of 50 assistant reference assessments

A fresh reviewer inspected 50 individual images using the same rubric, without primary labels, model scores or image statistics. The fixed hash selection comprises 10 development, 30 representative and 10 challenge cases. All original primary labels remain unchanged. This measures assistant repeatability, not agreement with human ratings.

| Panel | Images | Technical mean absolute difference | Aesthetic mean absolute difference | Repeat technical bias | Repeat aesthetic bias | Eligibility agreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.144 | 0.146 | -0.072 | -0.110 | 50/50 |
| development | 10 | 0.150 | 0.140 | -0.110 | -0.060 | 10/10 |
| representative | 30 | 0.150 | 0.143 | -0.043 | -0.110 | 30/30 |
| challenge | 10 | 0.120 | 0.160 | -0.120 | -0.160 | 10/10 |

There were no photography-floor decision disagreements and no per-task absolute differences of 0.75 points or more. Median absolute disagreement was 0.10 on both tasks; the 90th percentiles were 0.31 technical and 0.30 aesthetic. Repeats tended to score lower, especially for aesthetics (−0.11 on average). The small panel and assistant-only reviewers limit any claim about reference reliability. These estimates do not adjust or subtract from grader error.

Review elapsed time is recorded separately from HTTP latency, which is not applicable to assistant visual assessment. Each reference has the requested $0.00 price.
