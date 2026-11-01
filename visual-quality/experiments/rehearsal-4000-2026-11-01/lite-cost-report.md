## Q-ReAlign Lite infrastructure

All 4,000 Lite requests completed with valid technical and aesthetic scores. Their reported model price is **$0.00 per image**, as requested. This is an accounting override; Runpod compute and storage are accounted separately below.

The evaluation pod, `gm21jcse3yohgp` in EU-RO-1, ran from **13:46:11.489 to 15:16:43 UTC on 1 November 2026**. It used a secure RTX 4090 at a quoted **$0.89/hour**, with the unchanged scorer and checkpoint. The pod was stopped through the Runpod plugin after all responses were downloaded and their image, request and response hashes were verified.

An earlier replacement, `1e4rvgw2q4hkci`, incurred startup overhead from 13:42:39.928 to 13:44:24 UTC and received no image requests. Its posted charge was **$0.02587008**. The original pod could not resume because its host had no available GPU; that failed start created no new compute session.

At the latest **16:12:27.976 UTC** post-stop billing query:

- **Posted infrastructure subtotal: $1.29615666** across both replacements: **$1.27028659** for the evaluation pod, including reported storage, and **$0.02587008** for the startup-only pod. The evaluation pod's 13:00, 14:00 and 15:00 UTC billing buckets are now present. The provider did not explicitly confirm settlement, so this is a posted subtotal rather than a final invoice amount.
- **Elapsed-time compute estimate: $1.36851913**, calculated from both recorded lifecycle intervals at the quoted hourly rate. This estimate excludes storage and is not a settled invoice amount.
- **Settled infrastructure total: unavailable**, stored as null pending billing settlement. Neither the provisional subtotal nor the estimate is added to the user-assigned $0.00 model-price rows.

The earlier **15:17:49 UTC** posted subtotal of **$0.47511232** is retained with its original receipts in the infrastructure history. The [16:12:27.976 UTC refresh receipt](lite-billing-refresh-2026-11-01T16-12-27-976Z.json) records the new pod-specific billing response and confirms the original pod and both trial replacements remain **EXITED**; no evaluation GPU remains running. No pod was restarted or created during this check. Detailed provider receipts, audit timestamps and the distinction between posted charges and estimates are retained in [lite-infrastructure.json](lite-infrastructure.json). The complete request audit is in [lite-completion-validation.json](lite-completion-validation.json).
