# COMP-0484 AI Principles and Practice: Study Notes Index

Exam format: multiple choice, **5 questions per week** (Weeks 1–5 = 25 questions).

| Week | Topic | Notes file |
|---|---|---|
| 1 | Data Analysis & the ML Workflow | `Week 01 - Data Analysis Study Notes.md` |
| 2 | Supervised Learning (Regression & Classification) | `Week 02 - Supervised Learning Study Notes.md` |
| 3 | Ethics, Bias and Fairness | `Week 03 - Ethics Bias and Fairness Study Notes.md` |
| 4 | AI Safety, Governance and Explainability | `Week 04 - AI Safety Governance and Explainability Study Notes.md` |
| 5 | Unsupervised Learning & Representations | `Week 05 - Unsupervised Learning and Representations Study Notes.md` |

Each file has: **⭐ Important Highlights** (start here) → detailed notes → **✅ Quick self-check** (5 MCQ-style questions).

---

## Threads that run across the weeks (common exam angles)

1. **Leakage:** split first; fit every transform, imputer and augmentation (SMOTE) on training data only; cross-validate the *whole pipeline* (Weeks 1–2).
2. **Always look, never trust one number:** Anscombe's quartet (W1), residual plots behind R² (W2), accuracy hiding recall (W2), one fairness metric hiding others (W3), accuracy staying green under drift (W4), the average silhouette hiding fence-sitters (W5).
3. **Capacity / bias–variance:** polynomial degree, λ, tree depth, k in k-NN and number of trees are all one dial (W2).
4. **Rare positives:** accuracy and ROC-AUC flatter, so use precision/recall and PR-AUC (W2). Imbalance: try class weights first (W1 supplementary).
5. **Proxies:** proxy *labels* (W3 "who was approved before") → specification gaming (W4) → proxy *features* rebuilding the protected group:
   - **~70%** recoverable from the supervised score (W3/W4)
   - **71.1%** from PCA's PC1 (W5)
   - against a base rate of **~42%**
6. **Removing the sensitive column ≠ fairness** (redundant encoding). "Removed the word, not the information."
7. **Documentation is governance:** data dictionary + datasheet (W1, Gebru) → model card (Mitchell 2019) + system card (W4).
8. **The credit model's key numbers (W3–W5):**
   - Selection A 0.577 / B 0.215, so DI ratio **0.373** (< 0.80)
   - TPR 0.755 / 0.395
   - FPR 0.380 / 0.115
   - Precision 0.688 / 0.658
   - Base rates 0.52 / 0.37
   - Overall accuracy 0.70
   - Downturn: bad-loan rate 0.32 → 0.51 while accuracy and approval stay flat
9. **Judgement over automation:** choosing the threshold (W2), the fairness criterion (W3), and K (W5) are decisions you defend, not outputs of an algorithm.

## Key names → ideas (handy for MCQs)
| Name | Idea |
|---|---|
| Anscombe (1973) | Same stats, different data, so always plot |
| Chawla et al. (2002) | SMOTE |
| Gebru et al. | Datasheets for Datasets |
| Breiman (2001) | Random Forests; "Two Cultures" |
| Saito & Rehmsmeier (2015) | PR curve beats ROC on imbalanced data |
| Kleinberg, Mullainathan & Raghavan (2016); Chouldechova (2017) | Fairness impossibility result |
| Hardt, Price & Srebro (2016) | Equalised odds / equal opportunity |
| Buolamwini & Gebru (2018) | Gender Shades audit |
| Amodei et al. (2016) | Concrete Problems in AI Safety (spec gaming, reward hacking, distribution shift) |
| Mitchell et al. (2019) | Model cards |
| Ribeiro, Singh & Guestrin (2016) | LIME |
| Lundberg & Lee (2017) | SHAP |
| Rousseeuw (1987) | Silhouette |
| Mikolov et al. (2013) | Word embeddings (king − man + woman ≈ queen) |
| van der Maaten & Hinton (2008); McInnes et al. (2018) | t-SNE; UMAP |
| Shumailov et al. (2024) | Model collapse from recursive synthetic data |
