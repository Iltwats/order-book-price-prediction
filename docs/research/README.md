# Model improvement research evidence

Use the [synthesized report](../NEXT_MODEL_RESEARCH.md), [ordered action plan](../NEXT_ACTION_PLAN.md), and [300-word explanation](../NEXT_STEPS_SIMPLE.md) as the main deliverables. The action plan's shared trial cap governs the initial implementation; individual memos contain alternatives, not cumulative authorization for unrestricted sweeps.

| Approach | Research question | Evidence |
| --- | --- | --- |
| 1 | Which external weights satisfy the competition rules? | [Rules and provenance](01_rules_provenance.md) |
| 2 | Which genuinely small Hugging Face weights could fit? | [Small checkpoints](02_small_hf.md) |
| 3 | Can foundation models provide useful representations or teachers? | [Foundation checkpoints](03_foundation_hf.md) |
| 4 | Which order-book architectures transfer to anonymous features? | [LOB architectures](04_lob_architectures.md) |
| 5 | Can a better loss improve the existing head at unchanged inference cost? | [Metric-aligned readouts](05_metric_head.md) |
| 6 | How should the pretrained GRU be fine-tuned and exported? | [GRU fine-tuning](06_gru_finetuning.md) |
| 7 | Which alternative architectures support cheap causal streaming? | [Streaming architectures](07_streaming_architectures.md) |
| 8 | Which observable causal features could add information? | [Feature and data hypotheses](08_causal_features.md) |
| 9 | When do ensembles and compact students improve the tradeoff? | [Distillation and ensembles](09_distillation_ensembles.md) |
| 10 | Which validation mistakes could create an apparent improvement? | [Independent critical review](10_validation_red_team.md) |

The ten approaches were completed using three subagents in parallel batches; the session permits only three subagent threads alongside the main agent. Ten simultaneous independent subagents were not available.

## Structured inventories

- [MODEL_SHORTLIST.csv](MODEL_SHORTLIST.csv): 25 checkpoints or revisions, including actual file sizes where verified, model-card estimates where not, precision, license, provenance and disposition. This is research metadata, not competition data.
- [hf_small_inventory.json](hf_small_inventory.json): eleven exact checkpoint revisions with immutable commit IDs and source URLs. Branch tensor counts use metadata headers where the Hub summary repeated incorrect main-branch counts.
- [SOURCE_INDEX.csv](SOURCE_INDEX.csv): 109 unique external URLs cited in the ten memos. Different versions of the same paper or different pages of the same model count as different URLs, not independent studies. This is not a claim that every paper was read in full.
- [analysis_checks.json](analysis_checks.json): synthetic sufficient-statistic identity check, memory calculations and clearly labelled timing extrapolations. No competition data was evaluated for these calculations.

Evidence was checked on September 12, 2026. The ten subagent approaches report 41 Exa searches and 255 result slots; four additional synthesis searches returned 20 slots. Repeated result slots are not unique sources. Direct primary-source fetches, Hugging Face metadata requests and the separately disclosed fallback rule searches are additional verification steps. Substantive recommendations rely on publisher model cards, official files/code, author papers and local measured artifacts.

No external tensor weights, datasets or executable third-party code were adopted. No new competition model was trained, no new score improvement was measured, and no confirmation ledger was modified. The current ONNX model and submission archive remain unchanged. Provenance decisions for incomplete corpora are explicitly conditional rather than inferred from missing finance tags.
