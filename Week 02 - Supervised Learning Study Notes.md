# Week 2 — Supervised Learning: Regression and Classification

*COMP-0484 AI Principles and Practice · LOs: LO2, LO3*
*Sources: Week 02 Lecture Slides; Week 02 Reference Booklet*

---

## ⭐ Important Highlights (read these first)

1. **Supervised learning:** ŷ = f(x; θ). Features x go in, prediction ŷ comes out, and training tunes the parameters θ. **The goal is generalisation** (being right on unseen data), not fitting the training set. Memorising ≠ learning.
2. **Three splits:**
   - **Train** fits the parameters.
   - **Validation** is where you choose everything else (model, degree, k, λ).
   - **Test** is **spent once**, at the end.
   - Every peek at the test set to make a decision stops it being a test.
3. **Regression = predict a number. Classification = predict a category.** The better framing is that the target is a *decision you make*, not a fact of the data: "How much will they spend?" (regression) vs "Will they spend at all?" (classification).
4. **Linear regression** minimises the sum of squared errors and has a **closed-form** solution. It is the **honest baseline** to beat. Each βⱼ is the effect of one feature *holding the others fixed*.
5. **Polynomial degree is a capacity dial:** degree 1 underfits (train/test R² 0.53/0.51), degree 3 is about right (0.79/0.70), and degree 15 overfits (0.82/0.65: **train up, test down**).
6. **Ridge (L2, λΣβ²)** shrinks all coefficients smoothly and is good for collinear, noisy features. **Lasso (L1, λΣ|β|)** can set coefficients **exactly to zero**, so it does **feature selection**. λ is the bias–variance dial made explicit.
7. **Logistic regression is a CLASSIFIER**: P(y=1|x) = 1/(1+e^−(β₀+βᵀx)). It models the **log-odds** as linear and outputs a **probability**. The **threshold** you place on that probability is a separate decision.
8. **k-NN** has no training step. **k is the capacity dial:** k = 1 memorises (high variance), large k over-smooths (high bias). **Scale-sensitive**: standardise first.
9. **Decision trees** split for purity (**Gini or entropy**). They are the most interpretable model and among the **easiest to overfit**. **Depth** is the capacity dial (cap it or prune).
10. **Ensembles win by diversity, not brilliance:**
    - **Bagging / Random Forest** trains many trees on resampled data and feature subsets, then averages them. This **reduces variance**.
    - **Boosting / Gradient Boosting** adds small trees in sequence, each fixing the last one's errors. This **reduces bias**, but it is greedy and can overfit (XGBoost, LightGBM, HistGradientBoosting).
11. **SVM** finds the **widest margin**, and only the **support vectors** shape the boundary. **C** trades margin width against tolerating violations. The **kernel trick** gives non-linear boundaries. SVMs are good for clean margins, modest samples, and features ≫ rows (e.g. text). They are slow on huge data and give **distances, not calibrated probabilities**.
12. **Loss functions:** MSE punishes big misses hardest. MAE is kinder to outliers. **Log-loss punishes confident wrong answers**. "Which loss?" means "which mistakes can I least afford?"
13. **Bias–variance:** E[(y − f̂)²] = **Bias² + Variance + σ²** (σ² is irreducible noise). Total error is U-shaped, and the goal is the bottom of the U.
14. **Underfitting:** train and validation are both poor and close together, so add capacity. **Overfitting:** a wide train–validation gap, so reduce capacity or add data. They show the same symptom (bad validation score) but need **opposite cures**.
15. **Learning curves:** a gap that won't close means **variance** (more data, simplify, regularise). Both curves flat and low means **bias** (more features/capacity; **more data won't help**).
16. **Cross-validation:** k-fold rotates the held-out fold so every point is tested once, giving a **mean and a spread**. **Cross-validate the whole pipeline** so the scaler and imputer are refit per fold.
17. **Regression metrics:**
    - **RMSE** is in the target's units and punishes large errors.
    - **MAE** is the plain average error and robust to outliers.
    - **R² = 1 − SS_res/SS_tot** is the share of variance explained, unitless, and flatters on easy data.
    - A big **RMSE vs MAE gap** means a few large errors are doing the damage.
    - A high R² with **patterned residuals** is a warning.
18. **Accuracy lies on rare events:** "never fraud" at 3.4% prevalence scores **96.6% accuracy** and catches nothing.
19. **Precision = TP/(TP+FP)** ("of those flagged, how many were real": don't cry wolf). **Recall = TP/(TP+FN)** ("of the real ones, how many caught": don't miss). **F1** = harmonic mean. Don't optimise F1 out of habit, because it assumes both errors cost the same.
20. **ROC-AUC** = the probability the model ranks a random positive above a random negative (0.5 is a coin flip). It is threshold-free and good for ranking, but **says nothing about where to cut**.
21. **Rare positives → use the Precision–Recall curve / PR-AUC, not ROC-AUC.** ROC is flattered by the huge easy negative class.
22. **The threshold (0.5) is a convention, not a law:** use a **low** threshold for cancer screening (maximise recall) and a **high** one for blocking a card (protect precision). It is "a business decision wearing a maths costume."
23. **Framing chain:** Question → Target → Metric → Threshold. Get these right and plain logistic regression beats a mis-aimed neural net.
24. **Medical diagnosis means recall dominates. Spam/fraud means precision matters** (and PR curves). Same maths, opposite dial.

---

## 1. The supervised setup

- **Features x**: what we know about a case. **Target y**: what we want to predict. **Parameters θ**: what fitting adjusts.
- Fitting training data is trivial. Generalisation is the goal (the exam analogy: memorising past answers vs learning the method).
- Running example: the Week 1 customers table with two targets:
  - `monthly_charge` → regression
  - `churn` → classification (**base rate 15%**)
- Week 1's leakage-safe `prep` ColumnTransformer is reused as the first step of every model.

| Slice | Purpose |
|---|---|
| Training | Fit parameters |
| Validation | Choose model, degree, k, λ |
| Test | Final estimate, used once ("break glass in emergency") |

---

## 2. Regression

### Linear regression
ŷ = β₀ + β₁x₁ + … + βₚxₚ, fitted by min Σ(yᵢ − ŷᵢ)²
- Closed form, fast, deterministic.
- **Read the coefficients:** they are claims about the world. A negative income coefficient for charge means either the world is strange or your data is.

### Polynomial features
- Feeding x, x², x³ … into the same linear machinery. The model is still linear *in its weights*.

| Degree | Train R² | Test R² | Diagnosis |
|---|---|---|---|
| 1 | 0.53 | 0.51 | Underfit |
| 3 | 0.79 | 0.70 | About right |
| 15 | 0.82 | 0.65 | Overfit |

### Regularisation
- **Ridge:** min Σ(y−ŷ)² + λΣβⱼ² shrinks all coefficients toward zero and tames collinearity.
- **Lasso:** min Σ(y−ŷ)² + λΣ|βⱼ| can zero coefficients (selection, simpler story).
- On few clean features they barely differ: linear 0.668 / ridge 0.668 / lasso 0.670 R², with RMSE ≈ 4.7. **That sameness is the finding.** Regularisation earns its keep with **hundreds of correlated features**.

### Diagnosing a fit
- R² = 0.67 means ~two-thirds of the variation in charge is explained.
- **Residual plot:** you want a shapeless cloud around 0. A **curve** means a missed non-linearity. A **funnel** means the error grows with the prediction.

---

## 3. Classification

| Model | How it thinks | Capacity dial | Watch out |
|---|---|---|---|
| Logistic regression | Linear score → sigmoid → probability | regularisation | Linear boundary; you set the threshold |
| k-NN | Vote among k nearest | k (small = overfit) | Must standardise; lazy |
| Decision tree | Yes/no splits for purity (Gini/entropy) | depth | Overfits easily; most interpretable |
| Random forest (bagging) | Average many decorrelated trees | # trees, depth | Reduces **variance** |
| Gradient boosting | Sequential trees fix errors | learning rate, # trees | Reduces **bias**; can overfit |
| SVM | Widest-margin boundary; kernel trick | C, kernel | No calibrated probabilities; slow on big data |

**SVM objective:** min ½‖w‖² + C Σ max(0, 1 − yᵢ(wᵀxᵢ + b)) (hinge loss).

**Churn results (the trap):**

| Model | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Logistic | 0.880 | 0.600 | 0.652 | 0.625 | 0.949 |
| kNN (k=15) | 0.873 | 0.643 | **0.391** | 0.486 | 0.933 |
| Tree (d=4) | 0.900 | 0.643 | 0.783 | 0.706 | 0.931 |
| Forest | 0.887 | 0.636 | 0.609 | 0.622 | **0.956** |

All have ~88% accuracy, but k-NN **misses 6 in 10 churners**. Accuracy hides this.

**Decision boundaries:** logistic gives a straight cut, a tree carves rectangles, and a forest gives a smooth, flexible frontier. "Which model is best?" is the wrong question. Ask "best for what, measured how?"

---

## 4. Loss and the bias–variance trade-off

- **MSE** = (1/n)Σ(y−ŷ)² punishes large misses, so outliers dominate.
- **MAE** = (1/n)Σ|y−ŷ| is robust to outliers.
- **Log-loss** = −(1/n)Σ[y log p̂ + (1−y) log(1−p̂)] punishes confident wrong answers.

**Decomposition:** E[(y − f̂)²] = Bias² + Variance + σ²
- **Bias**: error from wrong or too-simple assumptions.
- **Variance**: the model jumps around as the training sample changes.
- **σ²**: the noise floor.
- As complexity rises, bias falls and variance rises.

**Every dial is the same dial:** polynomial degree, λ, tree depth, k, number of trees are all ways to slide along the bias–variance curve.

| | Underfitting | Overfitting |
|---|---|---|
| Train score | Poor | Excellent |
| Validation score | Poor (close to train) | Poor (big gap) |
| Cure | More capacity / features | Less capacity, regularise, more data |
| Learning curve | Both meet at a low ceiling | Persistent gap |

---

## 5. Evaluation that doesn't flatter

### Cross-validation
- 5-fold: train on 4 slices and score on the 5th. Every point is held out exactly once.
- Example: scores [0.958 0.942 0.892 0.925 0.883] give mean 0.920 and std 0.029. The single split said 0.88, so the spread shows how much one number wobbles.
- **Cross-validate the pipeline, not the model.** Otherwise the preprocessing leaks.

### Confusion matrix

|  | Predicted − | Predicted + |
|---|---|---|
| **Actual −** | TN | FP |
| **Actual +** | FN | TP |

Logistic churn model: TN 117, TP 15, FP 10, FN 8. **Decide which error hurts more before choosing a metric.**

### Metrics summary

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/all | Misleading on imbalance |
| Precision | TP/(TP+FP) | Of flagged, how many real |
| Recall (TPR, sensitivity) | TP/(TP+FN) | Of real, how many caught |
| F1 | 2PR/(P+R) | Harmonic mean; assumes equal costs |
| ROC-AUC | Area under TPR vs FPR | Ranking quality; threshold-free |
| PR-AUC | Area under precision vs recall | Honest for rare positives |
| RMSE | √((1/n)Σ(y−ŷ)²) | Target units; punishes large errors |
| MAE | (1/n)Σ\|y−ŷ\| | Robust |
| R² | 1 − Σ(y−ŷ)²/Σ(y−ȳ)² | Share of variance explained |

### Threshold
- Lowering the threshold raises recall and lowers precision.
- Cancer screen → low threshold. Card blocking → high threshold.

---

## 6. Framing real problems

**Question → Target → Metric → Threshold** (what decision? predict what? measure how? act when?)

| Problem | Frame | Metric the question demands |
|---|---|---|
| Demand forecasting, pricing | Regression | RMSE/MAE; asymmetric over/under-stock costs |
| Credit scoring | Classification | AUC + calibration; cost of default vs lost custom |
| Churn | Classification | Recall on likely leavers, or € value of retention |
| Medical diagnosis | Classification | **Recall first** (a missed case is the expensive error) |
| Fraud / spam | Classification | **Precision and PR-AUC** (rare positives) |
| Predictive maintenance | Classification | Recall with enough lead time |

**From prediction to decision:** a probability is not an action (e.g. 0.7 → phone them, 0.4 → send a discount). "The model informs; the policy decides." Data drifts, so thresholds rot, which closes Week 1's monitoring loop.

**Framing checklist:**
1. What decision does it feed, and what does a wrong one cost?
2. Is the honest target a number or a category?
3. Which error (FP or FN) hurts more?
4. Is the positive class rare? If so, retire accuracy and ROC for PR.
5. Where should the threshold sit so the costs balance?

**Key readings:** Breiman (2001) *Two Cultures* and *Random Forests*; Saito & Rehmsmeier (2015) on PR vs ROC; Provost, Fawcett & Kohavi (1998) *The Case Against Accuracy Estimation*.

---

## ✅ Quick self-check (MCQ style)

1. Training R² keeps rising while test R² falls as polynomial degree increases. This is: **(a)** underfitting **(b)** overfitting **(c)** irreducible error **(d)** leakage. → **b**
2. Which regulariser can set coefficients exactly to zero? **(a)** Ridge **(b)** Lasso **(c)** Both **(d)** Neither. → **b**
3. Learning curves show both train and validation scores converging at a low value. The best action is: **(a)** collect more data **(b)** add features/capacity **(c)** increase regularisation **(d)** reduce tree depth. → **b**
4. Fraud occurs in 3% of transactions. Which metric/curve is most informative? **(a)** Accuracy **(b)** ROC-AUC **(c)** Precision–Recall / PR-AUC **(d)** R². → **c**
5. Random forests primarily improve on a single tree by reducing: **(a)** bias **(b)** variance **(c)** irreducible noise **(d)** training time. → **b**
