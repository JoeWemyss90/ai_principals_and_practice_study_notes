# Week 3 — Ethics, Bias and Fairness

*COMP-0484 AI Principles and Practice · LOs: LO5, LO3*
*Sources: Week 03 Lecture Slides; Week 03 Reference Booklet*

---

## ⭐ Important Highlights (read these first)

1. **Fairness is a modelling decision, not a postscript.** It is decided upstream, in the choice of **target, loss and data**. A model is "an argument about the world".
2. **"Just maximise accuracy" is a value choice.** Accuracy treats every error the same and can hide concentrated harm on one group.
3. **Three ethical lenses:**
   - **Consequentialist**: best outcomes overall. Blind to how harm is distributed.
   - **Deontological**: fair rules, people as ends. May reject a model that helps the average person.
   - **Justice/Rawlsian**: improve the **worst-off**. May sacrifice aggregate performance.
   - Pick your lens **before** tuning.
4. **Three meanings of "bias":**
   - **Statistical**: an estimator is systematically off.
   - **Social**: a group gets systematically worse outcomes.
   - **Cognitive**: human heuristics skew judgement.
   - ⚠ **Reducing statistical bias can increase social bias**: a more accurate model of an unjust world reproduces the injustice more faithfully.
5. **Four types of data bias:**
   - **Sampling**: who's over- or under-represented.
   - **Historical**: past decisions baked in as ground truth.
   - **Measurement**: the recordable proxy ≠ the intended quantity.
   - **Labelling**: someone's judgement frozen as the target.
6. **The label is usually the real crime scene. Proxy label:** you wanted "creditworthiness" but have "did the bank approve them before and did they repay", so the model learns **who the old system favoured**.
7. **Algorithmic bias:** objectives reward shortcuts, **aggregation** optimises the average, and **regularisation** smooths minorities away (a minority looks like noise). "The optimiser is faithful, not malicious."
8. **Fairness through unawareness FAILS.** Deleting the sensitive column doesn't work because of **redundant encoding** (proxies such as income, credit history, defaults). In the example, group is recoverable from the score **70% of the time vs a 42% base rate**.
9. **Feedback loops:** deny loans to a group and you never observe their repayment, so the model grows more confident and more wrong. The disparity is **manufactured**, not just measured. Snapshot audits miss it.
10. **Group fairness** (compare rates across groups; easy; regulators use it) **vs individual fairness** ("similar people, similar treatment"; you must define "similar"). These conflict.
11. **Demographic (statistical) parity:** P(Ŷ=1|A) = P(Ŷ=1|B), equal selection rates. It ignores who deserved approval. Appropriate when you believe the **label itself is corrupted** by past injustice.
12. **Disparate impact / four-fifths rule:** the ratio of selection rates (disadvantaged ÷ advantaged) **< 0.80** is prima facie disparate impact. It is a legal test in **ratio space**. Example: 0.215/0.577 = **0.373**, which fails.
13. **Equalised odds:** **equal TPR AND equal FPR** across groups. It conditions on the truth. **Equal opportunity** is the relaxation that requires **equal TPR only**. Example: TPR 0.755 vs 0.395, which fails.
14. **Calibration:** a score of 0.7 means a 70% positive rate *within each group*. The example model is **calibrated and has near-equal precision (0.688 vs 0.658), yet is wildly unequal on selection and TPR.** Calibrated ≠ fair.
15. **One model, four verdicts:** fails parity, fails DI, fails equal opportunity, **passes** calibration/precision. "Is it fair?" is unanswerable. Ask "**fair by which criterion?**"
16. **Impossibility theorem (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017):** when **base rates differ**, you **cannot** simultaneously have equal calibration, equal FPR and equal FNR (except in degenerate cases).
17. **Every fix is a transfer:** forcing equal selection collapses Group B's precision from 0.66 to **0.50**. Unfairness is **moved, not removed**.
18. **Three places to intervene:**
    - **Pre-processing** (data: reweight, relabel, resample) touches the truth.
    - **In-processing** (training constraint) is the most principled but least portable.
    - **Post-processing** (per-group thresholds) is the easiest but means explicitly unequal treatment.
19. **Mitigation is redistribution:** post-processing for parity left aggregate accuracy almost unchanged, but **Group A's approvals fell 58% → 45%**. Always ask **who pays**, not just how much.
20. **Tools:** **Fairlearn** (Microsoft; lightweight, sklearn-native, `MetricFrame`) vs **AIF360** (IBM; 70+ metrics, many mitigations; research-grade).
21. **Ethical-impact assessment (5 parts):** Context → Findings (worst gap first) → Choice (criterion and why) → Cost (what was spent and who bore it) → **Residual risk** (what you'd refuse to ship).

---

## 1. Foundations: fairness is a modelling concern

- If ethics comes last, it is decoration. By shipping time every consequential choice is frozen.
- Every model encodes claims:
  - Choosing the **target** says what counts as success.
  - Choosing the **loss** says which mistakes you'll accept.
  - Choosing the **data** says whose past you generalise from.
- **Reframe:** don't ask "is this model biased?" (yes/no). Ask "**what does this model assume, and who benefits if the assumption holds?**"
- **Near zero-sum:** with a bounded approval budget, one threshold and one objective, a good fit for one group means a poor fit for another. It is not strictly zero-sum, since better data or features can lift both.
- **Whose fairness?** The applicant wants an equal shot, the lender wants precision on repayment, the regulator wants no disparate impact, and society wants the historical gap to shrink. **Power is the hidden variable**: whoever defines fairness usually isn't the person decided about.
- Old question: "How accurate is it?" Better question: "**Could you defend this decision to the person it was made about?**"
- Opening story: a bank deletes ethnicity, yet one group is approved at 58% and the other at 22%. The unfairness is **emergent** (data × objective × world), with no villain.

---

## 2. Where bias enters

| Sense | Means | Fix |
|---|---|---|
| Statistical | Estimator systematically off true value | More/better data, better model |
| Social | Systematically worse outcomes for a group | Often structural, not technical |
| Cognitive | Human judgement skewed by heuristics | Process, review, diverse teams |

**Data bias:** sampling, historical, measurement, labelling. Diagnose which is most likely and design a *test* for it rather than guessing.

**Proxy label:** a target that stands in for the true (unmeasurable) quantity. When the two diverge, the model learns the divergence. Audit the gap before computing any metric.

**Algorithmic bias:** objective shortcuts, aggregation (whose errors are visible), regularisation (smooths minorities).

**Societal bias:** some gaps are real *because* the world was unequal. Should a model reflect the world as it is or as it should be? There is no neutral answer, and **refusing to choose is a choice for the status quo**. Accuracy and justice are different axes.

**Pipeline:** bias enters at every stage, and the feedback loop turns today's decisions into tomorrow's "ground truth". When someone says "the algorithm is biased", make them point at the stage (usually upstream).

**Higher-order example:** a predictive-policing model can look more accurate every year while describing reality less well. It measures where police looked, not where crime is.

---

## 3. Fairness, made precise

**Method:** compute **one confusion matrix per group** and compare them.

| Group | TN | FP | FN | TP |
|---|---|---|---|---|
| A (n=1382) | 407 | 249 | 178 | 548 |
| B (n=1018) | 578 | 75 | 221 | 144 |

(Positive = approve; reality = would repay.)

### Worked calculations

| Quantity | Formula | Group A | Group B |
|---|---|---|---|
| Base rate (repays) | — | 0.52 | 0.37 |
| Selection rate | (TP+FP)/n | 797/1382 = **0.577** | 219/1018 = **0.215** |
| TPR | TP/(TP+FN) | 548/726 = **0.755** | 144/365 = **0.395** |
| FPR | FP/(FP+TN) | 249/656 = **0.380** | 75/653 = **0.115** |
| Precision (PPV) | TP/(TP+FP) | 548/797 = **0.688** | 144/219 = **0.658** |

**Disparate impact ratio** = 0.215 / 0.577 = **0.373** (< 0.80, fails).

### Criteria

| Criterion | Definition | Our model | Verdict |
|---|---|---|---|
| Demographic parity | Equal P(Ŷ=1) across groups | 0.577 vs 0.215 | ❌ Fails |
| Disparate impact (4/5 rule) | Ratio of selection rates ≥ 0.80 | 0.373 | ❌ Fails (legally exposed) |
| Equal opportunity | Equal TPR | 0.755 vs 0.395 | ❌ Fails |
| Equalised odds | Equal TPR **and** FPR | both differ | ❌ Fails |
| Calibration / precision | Score means the same per group | 0.688 vs 0.658 | ✅ Passes |

- **Demographic parity's blind spot:** it ignores who actually repays, so it can force approval of less-qualified applicants.
- **Why prefer equalised odds?** It conditions on the truth and doesn't demand equal approvals, only that being right or wrong doesn't depend on group.
- A qualified Group B applicant is **nearly half as likely** to be approved as an equally qualified Group A applicant.

---

## 4. Why you can't have it all

- **Impossibility result:** with different base rates, no classifier achieves equal calibration + equal FPR + equal FNR at once.
- Any tool promising a "fair" model, full stop, is "selling you a choice disguised as a solution".
- **Watching the trade:** force parity by lowering B's threshold until both select at 0.577. The selection gap closes, but **B's precision falls 0.658 → ~0.50**.
- **What algorithmic justice CAN do:** make trade-offs visible and quantified; force an explicit, auditable choice; catch disparities hidden by accuracy.
- **What it CAN'T do:** satisfy everyone; repair injustice that lives in the data; replace the **political** decision about which harm is worse.
- Choosing a criterion is a **stance on whether base-rate differences are real or the residue of injustice**. That decision belongs to stakeholders, ethicists and the public, not to the person who wrote the training script.

---

## 5. Lab: audit, mitigate, cost

```python
from fairlearn.metrics import MetricFrame, selection_rate
mf = MetricFrame(metrics={"selection": selection_rate,
                          "tpr": recall_score, "ppv": precision_score},
                 y_true=y_test, y_pred=y_hat, sensitive_features=group_test)
mf.by_group      # per-group table
mf.difference()  # selection 0.362, tpr 0.360, ppv 0.030
```

A 0.36 selection gap beside a 0.03 precision gap is "the whole ethical argument in three rows".

| Tool | Fairlearn (Microsoft) | AIF360 (IBM) |
|---|---|---|
| Feel | Lightweight, sklearn-native | Broad, research-grade |
| Metrics | `MetricFrame` per-group tables | 70+ metrics |
| Use when | You want a defensible answer fast | You want to compare many mitigations |

| Stage | Change | Trade-off |
|---|---|---|
| Pre-processing | Data (reweight/relabel/resample) | Touches truth; can feel like "cooking the books" |
| In-processing | Training (fairness constraint) | Most principled, least portable |
| Post-processing | Thresholds per group | Easiest; explicitly unequal treatment |

**Accuracy–fairness frontier** = a price list. Mitigation cost is usually **redistribution**, not accuracy.

| Box-tick audit | Strong audit |
|---|---|
| Reports one fairness number | Shows several metrics disagreeing |
| Declares the model "fair" | Names the criterion and defends it |
| Hides the trade-off | Quantifies what the fix costs |
| Treats mitigation as a fix | States what it would refuse to deploy |

### Five takeaways
1. Fairness is designed in, not inspected out.
2. Blindness isn't fairness (70% recoverable).
3. "Fair" is meaningless without a criterion.
4. Trade-offs are a theorem, not a bug.
5. Cost the fix and name who pays.

**Key readings:** Kleinberg et al. (2016); Chouldechova (2017); Hardt, Price & Srebro (2016), the origin of equalised odds; Barocas & Selbst (2016) *Big Data's Disparate Impact*; Buolamwini & Gebru (2018) *Gender Shades*; Barocas, Hardt & Narayanan, *Fairness and ML*; O'Neil, *Weapons of Math Destruction*.

### Glossary
- **Base rate**: the underlying proportion of a group with the positive outcome.
- **Calibration**: predicted probabilities match observed frequencies, ideally in every group.
- **Redundant encoding**: a sensitive attribute is present through correlated features.
- **Fairness through unawareness**: removing the sensitive attribute (inadequate).
- **Feedback loop**: the model's own decisions shape its future training data.

---

## ✅ Quick self-check (MCQ style)

1. Group A selection rate 0.60, Group B 0.42. Under the four-fifths rule: **(a)** passes (ratio 0.70) **(b)** fails (ratio 0.70 < 0.80) **(c)** passes (difference 0.18 < 0.20) **(d)** cannot tell. → **b**
2. Equal opportunity requires equal: **(a)** selection rates **(b)** true-positive rates **(c)** precision **(d)** TPR and FPR. → **b**
3. Removing the protected attribute from the inputs fails mainly because of: **(a)** overfitting **(b)** redundant encoding through proxies **(c)** class imbalance **(d)** calibration. → **b**
4. The impossibility result says that when base rates differ you cannot simultaneously have: **(a)** high accuracy and high AUC **(b)** equal calibration, equal FPR and equal FNR **(c)** parity and low variance **(d)** precision and recall above 0.5. → **b**
5. Applying different decision thresholds per group is which kind of mitigation? **(a)** pre-processing **(b)** in-processing **(c)** post-processing **(d)** data augmentation. → **c**
