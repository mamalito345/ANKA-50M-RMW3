# Experiment notebooks

These notebooks preserve the development history of ANKA-50M. They have been cleaned for publication: prose is in English, saved outputs and execution counters are removed, and each code cell is preceded by a short purpose and run note.

They are not one sequential `Run all` workflow. Several notebooks use historical Google Drive paths and create large datasets or checkpoints. Review each notebook's configuration cell before execution.

## Suggested reading order

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `ANKA_50M_C0_Throughput_Colab.ipynb` | Establishes the C0 baseline and A100 throughput measurement. |
| 2 | `ANKA_Data_and_Tokenizer_Pipeline.ipynb` | Builds and validates the earlier corpus and tokenizer pipeline. |
| 3 | `ANKA_C0_to_R2_M2_Experiments.ipynb` | Compares early C0, C1, C2, Reflect, and Morph variants. |
| 4 | `ANKA_R3_M3_Serious_Architecture_Training.ipynb` | Trains the more substantial standalone Reflect and Morph architectures. |
| 5 | `ANKA_R3_M3_Deep_Evaluation.ipynb` | Runs causal, loss, generation, and diagnostic tests on R3 and M3. |
| 6 | `ANKA_W3_Proof_Test.ipynb` | Tests the controlled WorldPairs data intervention. |
| 7 | `ANKA_Pairwise_Combinations_RAM_A100.ipynb` | Evaluates pairwise combinations with RAM-cached training data. |
| 8 | `ANKA_RMW3_Full_5B_Training_and_Evaluation.ipynb` | Defines, trains, audits, and evaluates the final RMW3 architecture. |

## Reproduction guidance

- Use a fresh Google Colab A100 runtime for a full training notebook.
- Mount Google Drive only when a notebook explicitly requests it.
- Update all `/content/drive/MyDrive/ANKA/...` paths for your account.
- Keep secrets in Colab Secrets; never paste access tokens into notebook cells.
- Run structural and smoke tests before starting a long training cell.
- Check free disk, system RAM, and GPU memory before building caches.
- Long training cells write resumable checkpoints; confirm their output directory before running.

The authoritative released checkpoint is the 5B base model linked from the repository README. Experimental Stage-2 runs are intentionally excluded from the release.
