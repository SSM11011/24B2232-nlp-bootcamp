# NLP Bootcamp Report: SciFact

Shibam Sundar Mahakud (24B2232). All numbers are on the validation claims.

## Q1. BM25 vs. dense (bge-large-en-v1.5)

**BM25 wins.**
- **1390** ("mcm5s2U … NNR codons"): BM25 ranks the gold doc 1st, dense 2nd.
- **673** ("LDL cholesterol … cardiovascular disease"): BM25 1st, dense 5th.

Both claims hinge on rare exact tokens (mcm5s2U, NNR, LDL-cholesterol) that BM25 matches literally. Dense embeddings blur them into near-synonyms.

**Dense wins.**
- **463**: "genomic aberrations of metastases" vs. the gold "mutational landscapes … metastasis". Dense ranks it 1st, BM25 10th.
- **1311**: "lncRNAs influence mRNA translation" vs. the gold "LincRNA-p21 suppresses … translation". Dense 1st, BM25 7th.

These are paraphrases with little word overlap, which dense models capture and BM25 misses.

## Q2. Largest Task 3 gain and its cost

Fusion gave the only gain. RRF(k=10) over BM25, bge-large and S-PubMedBert scored nDCG@10 0.9334, up from 0.9328 for bge-large alone (+0.0006). It costs about 22 ms per claim: 22.1 ms for the scientific encoder, 0.14 ms for BM25 and 0.08 ms for the fusion itself.

Reranking the hybrid's top 20 with MedCPT-Cross-Encoder *lowered* nDCG@10 to 0.913 (−0.020) and added 138 ms per claim (710 ms at k=100). The reranker itself works: it raises BM25 from 0.868 to 0.899 and S-PubMedBert from 0.834 to 0.894. bge-large is simply already stronger than it.

## Q3. Hard negatives and overfitting

Hard negatives helped. Validation nDCG@10 went from 0.9328 (base) to **0.9443** with hard negatives (BM25 plus dense, margin-filtered, top-1 skipped). Random negatives scored 0.9323, essentially no gain.

I evaluated validation nDCG@10 every 3 steps and kept the best checkpoint; `code/logs/task4_curves.png` shows both curves. With random negatives, training loss collapses toward 0 while validation falls steadily from step 3, which is overfitting. With hard negatives, validation peaks at step 15 and then drifts lower as loss keeps falling.

A first attempt (scale 20, lr 2e-5) overfit badly: 0.909 for random, 0.926 for hard. Run-to-run noise is about ±0.004.

## Q4. Share of verification errors from retrieval

2 of 45 erroneous validation claims are `retrieval_miss`, which is **4.4%**. The fine-tuned retriever places 114 of the 124 gold (claim, doc) pairs in its top 3, so most errors come from verification rather than retrieval. The breakdown is 17 other, 10 needs_multiple_docs, 9 entity_mismatch, 5 numerical, 2 negation and 2 retrieval_miss.

The tuned decision rule uses only the top-1 abstract. That favours precision (0.75) and costs recall on claims with several gold docs.

A hand spot-check of about 38 tags agreed in about 85% of cases. The misfires came from spelling variants such as "Dcr2" vs. "Dicer-2".

## Q5. Next step

I would fine-tune MedCPT-Cross-Encoder on the SciFact training claims, using hard negatives mined by our fine-tuned retriever with the same margin filter. The cross-encoder is the only component that hurt: −0.020 off the shelf, while it adds +0.03 to +0.06 on weaker first stages, so its ranking signal is real but miscalibrated for this domain.

A better top-1 also feeds Task 5 directly, because verification only reads the top-ranked abstract. Training costs minutes on about 440 pairs, and early stopping on validation guards against overfitting.
