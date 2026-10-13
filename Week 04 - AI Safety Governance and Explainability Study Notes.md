# Week 4 — AI Safety, Governance and Explainability

*COMP-0484 AI Principles and Practice · LOs: LO5, LO3*
*Sources: Week 04 Lecture Slides; Week 04 Reference Booklet*

---

## ⭐ Important Highlights (read these first)

1. **A safety failure is NOT a bug.** A bug crashes visibly, locally and in a contained way, and tests catch it. A safety failure is the **program running perfectly, doing exactly what was specified, and harming someone anyway**. "It passed all tests" is not a safety claim: tests cover the cases you thought of, and safety is about the ones you didn't.
2. **Specification vs intent:** you hand the model a **proxy** (target column, loss, dataset), never your intent. **Alignment** = how faithfully optimising the specification produces the intended outcome. The **gap** is everything the specification quietly assumes.
3. **Specification gaming:** satisfying the letter of the objective by an unintended route (the **boat-racing agent circling a lagoon** collecting points instead of finishing). **Reward hacking:** a sharper form where the shortcut scores well but means nothing (optimise clicks, get clickbait; optimise engagement, get outrage).
4. **Distribution shift is the default, not an edge case.**
   - **Covariate shift:** the *inputs* move (incomes fall, demographics change).
   - **Concept shift:** the *input→output relationship* moves (the same income now implies different repayment).
5. **Silent failure** is the mode to fear. In the downturn simulation **accuracy stayed flat (0.70 → 0.71), approval rate stayed flat (0.42 → 0.42), but the bad-loan rate jumped 0.32 → 0.51**, with no exception raised. **Monitor the metric that matters, and design monitoring BEFORE deployment.**
6. **Safety and capability are different axes.** A more capable model deserves **more** scrutiny, not less: capability buys reach, and reach multiplies the cost of error.
7. **Classify the failure by mechanism:**
   - Good offline metrics, falling real results, no errors → **silent failure under distribution shift**.
   - Aces the objective via one convenient feature → **specification gaming / reward hacking**.
   - Fine on tested inputs, bizarre just outside them → **brittleness / poor robustness**.
   - Fair by the metric, harms a group in practice → **a proxy label doing unseen work**.
8. **Governance = accountability, not paperwork:** who is responsible when the model harms someone, and how do they show they were careful? **Test:** could an outsider, handed your records, reconstruct who decided what and why?
9. **GDPR:**
   - **Article 22**: the right not to be subject to a decision based **solely** on automated processing with legal or similarly significant effect. "Solely" matters, since a human meaningfully in the loop changes the analysis.
   - **Data minimisation**: justify every feature.
   - The **contested right to explanation**.
   - A credit model is the **centre** of Article 22.
10. **EU AI Act sorts by HARM, not by technology. Four tiers:**
    - **Unacceptable** = banned (e.g. government social scoring).
    - **High-risk** = heavy obligations (**Annex III**, which includes **creditworthiness assessment**).
    - **Limited** = transparency duties (e.g. disclose you're talking to a bot).
    - **Minimal** = left alone (the vast majority).
11. **High-risk obligations:** risk-management system, data governance, technical documentation, record-keeping, **human oversight**, demonstrated accuracy and **robustness**, transparency to affected people. **The burden of proof flips:** you must demonstrate fitness *before* deployment.
12. **NIST AI RMF** (US, **voluntary**): four functions that **loop**, **Govern** (the wrapper: culture, roles, accountability), **Map** (context and risks), **Measure** (analyse and track), **Manage** (prioritise and respond).
13. **Three instruments:** **GDPR** = individual rights over personal data (empowers the **data subject**). **EU AI Act** = obligations proportional to harm (empowers the **regulator/public**). **NIST AI RMF** = voluntary continuous risk management (empowers the **organisation**). Tension: GDPR says collect less, while the Act says document more.
14. **Audit trail:** **data provenance**, **model lineage**, **decision logs**. These are written in advance or not at all. Retrofitting costs ~10×.
15. **Documentation artefacts:**
    - **Model card** (**Mitchell et al. 2019**): intended use, out-of-scope use, data, **metrics disaggregated by group**, ethical considerations, caveats.
    - **Datasheet** (**Gebru et al.**): why/by whom the data was collected, who's in it and who's missing, labelling, inappropriate uses.
    - **System card**: the model *in its deployed system* (thresholds, human override). The harm lives in the system.
16. **"None identified"** under ethical considerations is not a finding; it is a confession that nobody looked. The trust is in the **caveats**.
17. **Explainability, two axes:** **Global** (how the model behaves overall) vs **Local** (why *this* decision). **Intrinsic** (readable by construction: shallow tree, small linear model) vs **Post-hoc** (bolted onto a black box, i.e. "a model of a model").
18. **Feature importance:** **impurity** importance (trees only, fast, **biased toward high-cardinality features**) vs **permutation** importance (shuffle a feature and measure the damage; honest, but **splits credit unpredictably between correlated features**). They disagreed on the top feature (impurity: income; permutation: credit history).
19. **LIME** (Ribeiro et al. 2016): perturb around **one point**, fit a **local linear surrogate**. Model-agnostic, intuitive, per-decision. **Weakness:** the neighbourhood choice changes the story, making it **unstable**.
20. **SHAP** (Lundberg & Lee 2017): **Shapley values** from cooperative game theory, a feature's average marginal contribution over **all orderings**. Guarantees: **additive** (contributions sum to the prediction) and **consistent**. It is **expensive**. For a linear model φᵢ = wᵢ(xᵢ − E[xᵢ]).
21. **SHAP is honest about the MODEL, silent about the laundering:** "income drove this denial" can be true while income is a **proxy for the protected group**. An explanation is **not an exoneration**.
22. **Where explanations mislead:** **instability**, **correlation** (credit flips between correlated features), **plausibility bias** (neat ≠ true), **the alibi** (laundering a bad decision), **surrogate infidelity** (a 70%-fidelity surrogate is "a confident lie").
23. **Explanations make a model INTERROGABLE, not TRUSTWORTHY.** Use XAI as a weapon against your own model. Attack an explanation before you quote it.

---

## 1. Safety: the gap between what you asked for and what you got

Safety (in this module) = the **gap between what you asked a system to do and what it actually did**.

| Bug | Safety failure |
|---|---|
| Program stops, stack trace | Program runs perfectly |
| Visible, local, contained | Clean answer, confidently wrong |
| Tests catch it | Tests never sampled it |

### Specification and alignment
- **Intent:** approve applicants who will repay.
- **Specification:** maximise accuracy on "did the old system approve, and did they repay".
- **Gap:** this assumes old decisions were fair, that everyone who might have repaid got a chance, and that the past predicts the future.
- "Every spec is a translation, and every translation loses something."

### Gaming and hacking
- The credit model games its specification exactly like the boat: told to predict repayment, it learns "**who was approved before**".
- **Test for any objective:** name the cheapest way to satisfy it that you'd be embarrassed to defend. That is your reward-hacking risk, and it belongs in the governance note.

### Robustness
- "A model is a photograph of one moment, deployed into a film that keeps rolling."
- Robustness = resistance to shift. The **EU AI Act requires** it to be demonstrated for high-risk systems.

### Downturn simulation (concept shift)

| Quantity | Trained world | After downturn |
|---|---|---|
| Accuracy | 0.70 | 0.71 |
| Approval rate | 0.42 | 0.42 |
| **Bad-loan rate among approved** | **0.32** | **0.51** |
| Repayment rate (whole pool) | 0.46 | 0.29 |

You find out about silent failure "from a complaint, a regulator, a lawsuit, never from the model".

### Safety vs capability

| | More capable | Safer |
|---|---|---|
| Measured by | Accuracy, coverage, benchmarks | Failure rate on unseen data; behaviour under stress |
| Improves with | More data, bigger model, richer features | Constraints, monitoring, documentation, humility |
| Trap | A strong number hides a fragile system | A safe system can look under-optimised |

### Lifecycle, where safety is decided
Design (what to predict, about whom) → Data (which world to sample) → Training (what to optimise and what you forgot) → Evaluation (what to *stress*, not just score) → Deployment (what to monitor) → **Retirement** (when to stop trusting it). "The model did it" is never an answer, because a person chose each dial.

---

## 2. Governance: who answers for the decision

- If the documents can't locate responsibility after the fact, they are **theatre**.
- Governance is a **design constraint, not a final gate** (same move as fairness in Week 3).

### GDPR
- Not an AI law, but it bites AI hard.
- **Art. 22**: no solely automated decision with legal or similarly significant effect.
- **Data minimisation**: no hoarding features "just in case".
- **Right to explanation**: legally unsettled, trending toward *more*.

### EU AI Act tiers

| Tier | Treatment | Example |
|---|---|---|
| Unacceptable | Banned | Government social scoring |
| High-risk (Annex III) | Heavy obligations | **Credit scoring**, medical triage-type uses |
| Limited | Transparency duties | Chatbots must disclose they're bots |
| Minimal | Left alone | Spam filter, film recommender |

"A spam filter and a credit scorer are both 'AI'; only one can end a person's tenancy application."

### NIST AI RMF (AI 100-1)

| Function | Asks of you |
|---|---|
| **Govern** | Culture, roles, accountability around all the rest |
| **Map** | Establish the context and surface the risks |
| **Measure** | Analyse, benchmark, track risks over time |
| **Manage** | Prioritise, respond, allocate resources |

These are verbs that **loop** continuously while the model is live.

### Audit trail
- **Data provenance**: where each row came from, by what right.
- **Model lineage**: which data, code and version made this prediction.
- **Decision logs**: what the model output and who acted on it.

### Instruments compared

| Instrument | Organising idea | Empowers |
|---|---|---|
| GDPR | Individual rights over personal data | Data subject |
| EU AI Act | Obligations proportional to harm | Regulator & public |
| NIST AI RMF | Voluntary, continuous risk management | Organisation |

Satisfy the binding instruments and borrow structure from the voluntary one. Don't treat an adopted framework as if it were law.

---

## 3. Documentation

| Artefact | Answers | Origin |
|---|---|---|
| Model card | What the model is, and what it must never be used for | Mitchell et al. 2019 |
| Datasheet | What the data is, and who is missing | Gebru et al. 2018/2021 |
| System card | What happens when the model meets a real deployment | — |

"Most model failures are dataset failures wearing a model's clothes."

**Credit model card (honest version):**

| Section | Content |
|---|---|
| Intended use | Rank applicants by repayment probability **for a human reviewer** |
| Out of scope | Automatic denial with no human; populations not trained on |
| Metrics | 70% overall; **disaggregated selection 0.58 (A) vs 0.22 (B)** |
| Ethical considerations | Documented disparate impact, DI = 0.37 (Week 3 audit) |
| Caveats | Degrades under income shift; don't trust beyond **12 months** without re-validation |

| Box-tick card | Strong card |
|---|---|
| Names model, one accuracy figure | States out-of-scope use bluntly |
| "Intended for classification" | Disaggregates every metric by group |
| "Ethical considerations: none identified" | Links to the fairness audit |
| Silent on what breaks it | Names the shift that breaks it and when to stop trusting it |

---

## 4. Explainability (XAI)

| | Global | Local |
|---|---|---|
| Question | How does the model behave in general? | Why *this* decision for *this* person? |
| Tools | Feature importance, global SHAP (beeswarm) | LIME, local SHAP |

**Intrinsic vs post-hoc:** everything below is post-hoc, "a fallible witness, not a confession".

### Feature importance (random forest on credit data)

| Feature | Impurity | Permutation |
|---|---|---|
| income | **0.36** | 0.04 |
| credit history | 0.32 | **0.04** |
| debt ratio | 0.27 | 0.00 |
| prior defaults | 0.05 | 0.00 |

Read it as a **hint, never a ranking you'd defend in a tribunal**.

### LIME
- Perturb the inputs around one point, watch the black box, fit a simple linear model locally, and read off the coefficients.
- The neighbourhood (width and weighting) is a choice, so two analysts can get different, equally confident explanations.
- It is fairly stable for linear models. **Instability grows with non-linearity** (trees, networks).

### SHAP
φᵢ = Σ over S ⊆ F∖{i} of [|S|!(|F|−|S|−1)!/|F|!] · [v(S∪{i}) − v(S)]
- Each feature is a "player"; the prediction is the payoff.
- **Additive**: contributions sum exactly to the prediction. **Consistent**: a feature that never helps gets no credit.
- Global SHAP (mean |value|, log-odds): income 0.48, credit history 0.38, debt ratio 0.21, prior defaults 0.05.
- Local: a denied Group B applicant (p ≈ 0.03) was pushed down by income, then debt ratio, then credit history.
- But **income was built partly from group** (baseline minus several thousand euro for Group B), so SHAP faithfully reports a proxy.

### Where explanations mislead
| Failure | What happens |
|---|---|
| Instability | Re-run LIME → different story |
| Correlation | Credit flips between correlated features |
| Plausibility bias | Neat feels true; neat ≠ true |
| The alibi | Explanation launders a bad decision |
| Surrogate infidelity | Must report **fidelity**; 70% surrogate = confident lie |

| An explanation buys you | It does NOT buy you |
|---|---|
| A place to point; a hypothesis; a sharper next question; **interrogability** | Proof the model is right; proof the stated reason is real; permission to stop checking; **trustworthiness** |

---

## 5. Lab: explain, break, document

1. **Explain:** SHAP (global) and LIME (local). Ask which drivers are **proxies wearing a respectable name**.
2. **Break:** impose a distribution shift and measure the drop (no error is raised).
3. **Document:** a model card and a one-page governance note.

| Box-tick governance note | Strong governance note |
|---|---|
| "Explainable (SHAP applied)" | Shows where SHAP and LIME disagree |
| "GDPR compliant" | Names the EU AI Act tier and obligations |
| "No significant risks" | Names the shift that breaks it and the monitoring that catches it |
| Ends there | States what it would refuse to deploy without a human in the loop |

### Five takeaways
1. A safety failure is the model working as specified and harming anyway.
2. Governance is accountability you can audit.
3. The EU AI Act sorts by harm. Credit scoring is high-risk, which is a to-do list carried *before* deployment.
4. Documentation is the interface for oversight. "None identified" means nobody looked.
5. An explanation makes a model interrogable, not trustworthy.

**Key readings:** **Amodei et al. (2016) *Concrete Problems in AI Safety*** (failure-mode taxonomy); Mitchell et al. (2019) model cards; Gebru et al. (2021) datasheets; Ribeiro et al. (2016) LIME; Lundberg & Lee (2017) SHAP; Molnar *Interpretable ML*; NIST AI RMF; EU AI Act Annex III; GDPR Art. 22.

---

## ✅ Quick self-check (MCQ style)

1. A recession makes the same income imply a lower repayment probability. This is: **(a)** covariate shift **(b)** concept shift **(c)** reward hacking **(d)** label noise. → **b**
2. Under the EU AI Act, a creditworthiness-assessment system is classed as: **(a)** unacceptable **(b)** high-risk **(c)** limited **(d)** minimal. → **b**
3. Which NIST AI RMF function covers culture, roles and accountability? **(a)** Map **(b)** Measure **(c)** Manage **(d)** Govern. → **d**
4. Which explanation method has the guarantee that feature contributions sum exactly to the prediction? **(a)** LIME **(b)** impurity importance **(c)** SHAP **(d)** permutation importance. → **c**
5. GDPR Article 22 concerns: **(a)** data portability **(b)** the right not to be subject to solely automated decisions with significant effects **(c)** the right to be forgotten **(d)** breach notification. → **b**
