# Data and tokenizer

ANKA's 16,384-token English tokenizer was trained from scratch. Validation reported zero unknown tokens and zero round-trip failures on the sampled validation corpus.

The 5B-token training mixture combined educational web text, mathematics and science text, academic and biomedical material, reference text, books, educational material, question-answer content, and WorldPairs. Source preparation and mixture construction are documented as executable code in:

- `notebooks/ANKA_Data_and_Tokenizer_Pipeline.ipynb`
- `notebooks/ANKA_W3_Proof_Test.ipynb`
- `notebooks/ANKA_RMW3_Full_5B_Training_and_Evaluation.ipynb`

The repository does not redistribute the full training corpus. Every upstream dataset remains subject to its own license and terms.
