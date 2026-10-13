# Week 1 — Data Analysis and the ML Workflow

*COMP-0484 AI Principles and Practice · LOs: LO2, LO5*
*Sources: Week 01 Lecture Slides; Week 01 Supplementary Notes (the supplementary notes say they are "not assessed directly". Their key points are summarised at the end)*

---

## ⭐ Important Highlights (read these first)

1. **The ML workflow is a loop, not a line:** Frame → Collect → EDA → Preprocess → Model → Evaluate → Deploy → Monitor, and monitoring feeds back into reframing. Weeks 1–5 mostly cover stages 1–4.
2. **Variable type decides everything:** it sets the summary statistic, the plot, the encoding and the model.
   - Binary (churn 0/1), Nominal (plan, region), **Ordinal** (satisfaction: ordered, unequal gaps), Discrete (tenure), Continuous (age, income).
3. **Encoding:** use **ordinal encoding** only when a real order exists (Low→0, Med→1, High→2). Use **one-hot** for nominal variables, so no false order is introduced.
4. **Mean vs median:** the mean chases outliers and the median doesn't. **Mean ≫ median means right skew/outliers** (income: mean 53,565 vs median 40,388).
5. **IQR = Q3 − Q1** is the robust measure of spread. SD is in the data's units but is pulled by outliers.
6. **Skewness g₁** measures asymmetry. **Excess kurtosis g₂** measures tailedness (the formula subtracts 3, so a Normal = 0).
7. **Box plot:** box = IQR, line = median, whiskers reach **1.5 × IQR**, and points beyond them are flagged.
8. **Outlier rules:** |z| > 3 (assumes roughly Normal data) **or** the IQR fences x < Q1 − 1.5·IQR / x > Q3 + 1.5·IQR (no Normality assumption).
9. **Pearson r** measures **linear** association only, r ∈ [−1, 1], and is wrecked by outliers. **Spearman ρ** is Pearson on **ranks**: it measures **monotonic** association and is robust to outliers and curves. Example: same columns, Pearson 0.14 vs Spearman 0.82.
10. **Anscombe's quartet:** four datasets with the same mean, variance, correlation and regression line, but very different plots. **Always plot.**
11. **Correlation ≠ causation.** A correlation can't see curves, clusters or a lurking third variable.
12. **Missing data mechanisms:**
    - **MCAR**: missing completely at random. Gaps are unrelated to anything, so dropping rows is "safe-ish".
    - **MAR**: missingness depends on *observed* features (e.g. younger users skip income).
    - **MNAR**: missingness depends on the *unobserved value itself* (high earners hide income).
    - "Missing" is often signal, so add a was-missing flag.
13. **Imputation:** median fill is robust but **creates a spike** at one value and **distorts variance**. KNN/iterative imputation uses the other columns and spreads values more realistically. **Fit the imputer on training data only.**
14. **Scaling:** z-score (x−μ)/σ; min–max (x−min)/(max−min). **Distance- and gradient-based models** need comparable scales.
15. **Augmentation rules:** transformations must be **label-preserving** and **realistic** (flipping a "6" makes a "9", which is a bug). **Split first, augment the TRAINING set only. Never augment validation/test.**
16. **SMOTE:** x_new = xᵢ + λ(x_zᵢ − xᵢ), λ ~ U(0,1). It interpolates between a minority point and a minority nearest neighbour. Example: {0:382, 1:68} → {0:382, 1:382}.
17. **Leakage is the cardinal sin:** any statistic learned from the *whole* dataset before the split (scaling, imputation, feature selection, encoding, augmentation) leaks test information.
    - Demo: feature selection on pure noise *outside* the pipeline gives **0.795** accuracy, which is a lie. *Inside* the pipeline it gives **0.510** (chance).
    - Fix: **split first → fit transforms on the training fold only → wrap everything in one Pipeline → cross-validate the whole pipeline.**
18. **Batch vs stream:** batch means a fixed dataset that is reproducible and the default for training. Stream means data arrives continuously, with bounded memory, and you must handle drift and online updates.
19. **Documentation is governance (LO5):** a **data dictionary** describes every field (type, units, missingness). A **datasheet** (Gebru et al.) describes the dataset: why and how it was collected, consent, who is represented or missing, licence, and known biases.

---

## 1. The end-to-end ML workflow

```
Frame → Collect → EDA → Preprocess
                              ↓
Monitor ← Deploy ← Evaluate ← Model     (and Monitor feeds back to Frame)
```

| Workflow stage | Course week |
|---|---|
| Data, EDA, preprocessing | Week 1 |
| Modelling (supervised) | Week 2 |
| Ethics, fairness, safety, governance | Weeks 3–4 |
| Representations / unsupervised | Week 5 |
| Neural networks & maths | Weeks 7–8 |
| Transfer, transformers, diffusion | Weeks 9–11 |

**Reproducibility:** fixed random seed, one script (`make_assets.py`) reproduces every figure.

**Profile before you model:** `df.describe()`. In the running example, income std = 112,607 (huge: outliers and skew), and median ≪ mean.

---

## 2. Data types and encoding

| Type | What it is | Example |
|---|---|---|
| Binary | Exactly two categories | churn 0/1 |
| Categorical (nominal) | Unordered groups | region, plan |
| Ordinal | Ordered, unequal gaps | satisfaction |
| Discrete | Countable integers | tenure (months) |
| Continuous | Any value in a range | age, income |

- Models need numbers, so categories must be encoded.
- **OrdinalEncoder** (`categories=[["Low","Medium","High"]]`) is a monotone map that preserves order. Only valid if a real order exists.
- **OneHotEncoder** creates one column per category with exactly one "hot". Use it for nominal variables. Use `handle_unknown="ignore"` in pipelines.

---

## 3. Summary statistics

| Measure | Formula / meaning | Robust? |
|---|---|---|
| Mean | x̄ = (1/n) Σxᵢ | ❌ chases outliers |
| Median | middle of sorted values | ✅ |
| Variance | s² = (1/(n−1)) Σ(xᵢ − x̄)² | ❌ |
| SD | s = √s², in the data's units | ❌ |
| IQR | Q3 − Q1 | ✅ |
| Skewness g₁ | asymmetry (3rd moment / s³) | — |
| Excess kurtosis g₂ | tailedness (4th moment / s⁴ − 3) | — |

Example values (income): skew = 10.99 (very right-skewed), kurtosis = 122.93 (extremely heavy tails). IQR = 23,204.5 (Q1 = 30,143, Q3 = 53,348).

---

## 4. Distributions and visualisation

- Normal pdf: f(x) = (1/(σ√2π)) exp(−(x−μ)²/2σ²). It is fully described by μ and σ, **but most real data is not Normal**.
- Two datasets with the **same mean and SD** can be unimodal vs bimodal, so **always plot**.

| Question | Chart |
|---|---|
| Distribution of one variable | Histogram, KDE, box/violin |
| Relationship between two | Scatter, 2-D density |
| Comparison across groups | Grouped bar, box-by-category |
| Trend over time | Line chart |
| Many variables at once | Correlation heatmap, pair plot |

"Choose the chart for the question; label honestly; never let a default colour map decide the story."

---

## 5. Correlation and its traps

- **Pearson:** r = Σ(xᵢ−x̄)(yᵢ−ȳ) / (√Σ(xᵢ−x̄)² · √Σ(yᵢ−ȳ)²). It measures **linear** association only.
- **Spearman:** ρ = Pearson(rank(x), rank(y)). It measures **monotonic** association and is immune to outliers and curves.
- Example: income vs monthly_charge gives Pearson 0.144 (p = 9.6e-4) but Spearman 0.821. A few extreme incomes wrecked Pearson.
- **Anscombe's quartet** has identical summary statistics and regression line but completely different data.
- `df.corr()` uses Pearson by default and silently drops non-numeric columns.
- **Correlation ≠ causation.**

---

## 6. Data processing

### Batch vs stream
| Batch | Stream |
|---|---|
| Fixed dataset, processed together | Data arrives continuously |
| Reproducible, easy to validate | Bounded memory; must handle drift |
| Default for training/coursework | Statistics/models update online |

### Feature scaling
- **z-score (StandardScaler):** z = (x − μ)/σ
- **min–max:** x′ = (x − x_min)/(x_max − x_min)
- Needed by **distance-based** (k-NN, k-means, SVM) and **gradient-based** models.

### Missing data
- Check with `df.isna().sum()`. Look for a *pattern*, not just the count.
- MCAR / MAR / MNAR (see Highlights). The mechanism decides whether dropping or imputing is honest.
- `SimpleImputer(strategy="median")`: missing 53 → 0 (filled with 40,388).
- ⚠ Median fill stacks every gap on one value, creating a spike that distorts variance. KNN imputation is more realistic.
- ⚠ Fit the imputer on training data only.

### Outliers
- z-rule |z| > 3 (assumes Normal) vs IQR rule (no assumption). Example: upper fence = 88,154, 26 flagged, max = 1,342,344.
- Decide per cause: **error**, **rare-but-real**, or **different population**.

### When data is not enough
Too few examples leads to overfitting and high variance, and rare classes vanish (churn = 91/600 = 15%).
Options, **cheapest last**: collect more/better data (best but slow) → regularise or use a simpler model → transfer learning → data augmentation.

### Augmentation by data type
| Data | Label-preserving transforms |
|---|---|
| Images | flip, rotate, crop, scale, colour jitter, noise, cut-out |
| Text | synonym swap, back-translation, random insert/delete/swap |
| Tabular | SMOTE, Gaussian jitter, resample a fitted model |
| Audio/time-series | time-shift, pitch-shift, noise, time-warp |
| Generative (any) | VAEs, GANs, diffusion: learn the distribution, then sample |

### Generative augmentation
| Method | Idea |
|---|---|
| VAE | Encode to a smooth latent space, sample, decode |
| GAN | Generator vs discriminator (tables via CTGAN) |
| Diffusion | Denoise random noise into a sample; best for images |

Ready-made tools: CTGAN/TVAE (SDV library) for tables, and LLM paraphrasing for text. ⚠ Generators need training data, can **amplify bias**, may invent unrealistic rows, and must be **fit on the training split only**.

### Augment the training set only
- Augmenting the test set means you evaluate on synthetic data.
- Fitting SMOTE before the split means synthetic neighbours cross the split, which is leakage.
- Put augmentation inside the pipeline so it only sees training folds.

---

## 7. Leakage-safe pipelines

**Leakage:** any statistic learned from the whole dataset before the train/test split lets test information seep into training. Scores look great, then collapse in production.

Demo (200 samples, 5000 noise columns, random labels):
- `SelectKBest` on ALL data, then CV gives **0.795** (impossible, a lie)
- `SelectKBest` *inside* the Pipeline gives **0.510** (chance, the truth)

Reusable template:
```python
prep = ColumnTransformer([
  ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())]), num),
  ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)   # after train_test_split FIRST
```

---

## 8. Governance and documentation

- **Data dictionary:** per field, giving type, units, missingness and notes (e.g. "income: continuous (€), skewed, 8.8% missing"; "satisfaction: ordinal Low<Med<High"; "churn: binary target").
- **Datasheet** (Gebru et al., *Datasheets for Datasets*): why/how collected, by whom, consent; who is represented and who is missing; licence and appropriate use; known biases and limitations.
- The biases studied in Weeks 3–4 are usually **baked in at collection time**, before any model exists.

---

## 9. Supplementary notes (not directly assessed, but useful context)

**Correlation in depth**
- Covariance's units are the product of x's and y's units. Pearson divides by both SDs, so it is unitless and bounded.
- **r = cosine of the angle** between the mean-centred columns, which is why it only sees straight lines.
- **r²** = share of variance explained by a straight-line fit. r = 0.5 means r² = 0.25, so 75% is unexplained.
- Pearson's assumptions: linearity and no influential outliers (both always matter); homoscedasticity (matters for CI/p-value); bivariate normality (only needed for the classical significance test).
- **Fisher transform:** z = arctanh(r), SE ≈ 1/√(n−3). Use it for confidence intervals.
- **Significance ≠ strength:** with n = 10,000, r = 0.03 is significant but trivial.
- **Kendall's τ** works on concordant/discordant pairs: τ = (C − D)/C(n,2). (τ+1)/2 = P(a pair is concordant). |τ| is usually smaller than |ρ|. Prefer it for small n or many ties.
- **Diagnostic:** |ρ| ≫ |r| means skew, outliers or curvature. |r| ≫ |ρ| means a few extreme points are manufacturing a line.
- Mixed types: point-biserial (binary–continuous), phi φ (binary–binary), Cramér's V (nominal–nominal, symmetric), Theil's U (asymmetric), correlation ratio η (nominal–continuous).
- **Partial correlation** controls for a third variable (ice cream vs drownings, controlling for temperature). ⚠ Controlling for a **collider** can *create* a spurious correlation.
- **Simpson's paradox:** the pooled correlation can have the opposite sign to every subgroup. **Ecological fallacy:** aggregate correlations don't transfer to individuals.
- **Restriction of range** (e.g. only admitted students) shrinks correlation. **Attenuation:** measurement error biases r toward 0.
- **Multiple testing:** 20 columns give 190 tests, so expect ~9–10 false positives at 5%. Use Benjamini–Hochberg.
- **Time series:** two random walks can show spurious correlation, so difference the series first.
- **Modern measures** that are 0 *iff* the variables are independent: distance correlation, Chatterjee's ξ (asymmetric, detects functional relationships), mutual information, HSIC. MIC is a cautionary tale (low power).
- **Five explanations for an association:** x→y, y→x, a confounder z, selection/collider bias, or chance.

**Augmentation in depth**
- Why it works: the **invariance** argument, **vicinal risk minimisation**, and **regularisation** (reduces variance).
- **Two rules:** (1) preserve the label; (2) produce inputs that could plausibly occur at deployment.
- Images: random resized crop is the most valuable transform. Also Cutout/Random Erasing, **Mixup** (blend inputs and labels, λ ~ Beta), CutMix, and RandAugment/TrivialAugment (simple and as good as searched policies).
- Text: EDA (synonym/insert/swap/delete), back-translation, masked-LM substitution, LLM generation (hazards: bias, diversity collapse, licence, wrong labels), and **counterfactual augmentation for fairness** ("he is a nurse" → "she is a nurse").
- Tabular is hardest. SMOTE variants: Borderline-SMOTE, ADASYN, SMOTE-NC (mixed types), SMOTETomek/ENN. SMOTE is weak in high dimensions and can't grow the minority convex hull.
- **Try class weighting first** (`class_weight="balanced"`). It often matches SMOTE with no leakage risk.
- ⚠ sklearn's `Pipeline` can't hold a resampler. Use **`imblearn.pipeline.Pipeline`**, which resamples during fit only.
- Leakage demo: SMOTE before CV on random labels gives ROC-AUC **0.99**. Inside an imblearn pipeline it gives **0.47**.
- Test set must reflect the **real deployment distribution**, including imbalance.
- **Model collapse** (Shumailov 2024): training recursively on generated data loses the tails of the distribution.
- **Test-time augmentation** is OK: it averages predictions over transformed inputs and is not training on the test set.
- Synthetic data is **not automatically anonymous** under GDPR.
- Decision order for scarce data: more real data → check metric/threshold → class weights / simpler model → transfer learning → simple augmentation → generative augmentation last.

---

## ✅ Quick self-check (MCQ style)

1. Income has mean 53,565 and median 40,388. This most likely indicates: **(a)** left skew **(b)** right skew/outliers **(c)** a Normal distribution **(d)** missing data. → **b**
2. High earners refusing to report income is an example of: **(a)** MCAR **(b)** MAR **(c)** MNAR **(d)** leakage. → **c**
3. Pearson = 0.14 but Spearman = 0.82 on the same columns. The best explanation is: **(a)** the relationship is negative **(b)** outliers/non-linearity are distorting Pearson **(c)** the sample is too small **(d)** the variables are independent. → **b**
4. Which is the correct order to avoid leakage? **(a)** scale → split → fit **(b)** SMOTE → split → fit **(c)** split → fit transforms on train only → evaluate on test **(d)** impute on all data → split. → **c**
5. Satisfaction (Low/Medium/High) should usually be encoded with: **(a)** one-hot **(b)** ordinal encoding **(c)** min–max scaling **(d)** SMOTE. → **b**
