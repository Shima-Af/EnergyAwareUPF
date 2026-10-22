# Scientific audit archive — 22 October 2026

[Read the scientific repository audit](SCIENTIFIC_REPOSITORY_AUDIT.md) and then the [submitted-manuscript addendum](submitted-manuscript-reconciliation/AUDIT_ADDENDUM.md). The addendum takes precedence for claims about the submitted paper.

## Saved analyses and evidence

- [Original audit](SCIENTIFIC_REPOSITORY_AUDIT.md): inventory, reproduction checks, all 13 reviewer assessments, scientific risks and proposed recovery plan.
- [Audit addendum](submitted-manuscript-reconciliation/AUDIT_ADDENDUM.md): corrections after the author supplied the submitted TeX.
- [Policy replay results](replay_results.json), [baseline results](baseline_results.json), and [profiling residuals](residual_results.json).
- [Historical artifact hashes](artifact_manifest.json), inference harnesses, logs, and seed-42 replay CSVs.
- [Submitted manuscript snapshot](submitted-manuscript-reconciliation/submitted_reference.tex), [comparison with the previous repository manuscript](submitted-manuscript-reconciliation/repository_vs_submitted.diff), and the previous manuscript backup.
- [Archive integrity manifest](archive_manifest.json): source and saved hashes for the copied files.

## Interpretation and provenance

These files preserve an audit of commit `a23e7f7de6c6e07fddad8411dde542b38123e785`. They are historical findings, not a claim that every later workspace file still has its audited hash. After the audit, the author supplied the submitted manuscript and explicitly requested that the repository TeX match it; that synchronization was completed and verified byte for byte. References to the old repository manuscript describe its state before that synchronization.

The archive was copied from `/tmp/upf-scientific-audit/`. Only Markdown link targets pointing into that temporary directory were redirected to the saved archive; analytical content and experimental evidence were preserved. The historical manifests retain their original paths and hashes. The new archive manifest records the saved copies separately.

The inference harnesses retain their original paths and output destinations. Archiving them did not execute them or start any new experiments. No reviewer fixes or retraining are included in this archive. Git commits have not been created.
