# Blind review of controlled degradation families

Run this only after the final 4,000-image manifest exists. The generator reads that manifest and the 400 exact delivered family views, never grader outputs, reference scores or image-stat text. It creates ten packets of ten families; each family has a parent and three ordered derivatives.

```sh
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/rehearsal-4000-2026-11-01/controlled_review.py generate
```

Packet JSON, review templates and the index are written to the experiment's `controlled-review/` directory. Lossless contact sheets and individual native crops are written under `/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/controlled-review-assets/`, keeping large review assets outside tracked experiment metadata. Existing destinations are protected from overwrite.

Each contact sheet contains the complete frame and one crop for every member. The crop is chosen from the parent alone using a fixed grid of detail candidates. Its coordinates are applied unchanged to all four images, with no crop resampling or enhancement. A small image uses a correspondingly smaller crop. Open the separate crop PNG to inspect native pixels if the image viewer scales the contact sheet. The packet also links the exact original delivered views and records source/view hashes. Full transform parameters, canonical-parent lineage and crop-selection provenance are in the separate `provenance.json`; reviewers should leave that file closed until their decisions are saved.

Review every family and all three derivatives. For each derivative record both `versus_parent` and `versus_previous`, using `clear_degradation`, `not_clearly_worse` or `unjudgeable`, with brief visible evidence. For V1, the preceding level is the parent, so both entries must agree. Inspect full-frame previews and native crops; open individual delivered images when necessary and list their case IDs in `individual_views_opened`. Generation order does not establish worsening. Keep imperceptible, improved and ambiguous cases in the dataset and use the appropriate classification. These are categorical stimulus checks, not additional numeric reference scores.

Save a completed copy of each review template and validate it:

```sh
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/rehearsal-4000-2026-11-01/controlled_review.py validate \
  --packet visual-quality/experiments/rehearsal-4000-2026-11-01/controlled-review/controlled-01.json \
  --reviews /absolute/path/to/completed-controlled-01.json
```

Validation requires all ten families, every derivative, valid identities, unchanged views/crops/contact sheets, a reviewer name and consistent V1 comparisons. It does not impose a preferred judgment.

Report all-family model diagnostics first. Separately report the comparisons visually verified against their parent, the severe derivative versus parent subset, and the families whose three adjacent comparisons are all verified. There are 600 recorded comparison entries but 500 unique image pairs because V1's parent and previous comparisons are identical. Never use the verified subsets to erase the original results or silently discard families.
