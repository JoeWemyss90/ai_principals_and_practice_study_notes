# Week 5 — Unsupervised Learning and the Idea of a Representation

*COMP-0484 AI Principles and Practice · LOs: LO1, LO2*
*Sources: Week 05 Lecture Slides; Week 05 Reference Booklet*

---

## ⭐ Important Highlights (read these first)

1. **Unsupervised = inputs only, no labels.** There is no accuracy, no test set and no answer key. **Clustering PROPOSES/IMPOSES structure; it does not "discover" the true groups.** The output is a **hypothesis you must defend**.
2. **K-means (Lloyd's algorithm):** pick K centres → **assign** each point to its nearest centroid → **update** each centroid to the mean of its members → repeat. It minimises **within-cluster inertia** W = Σ_c Σ_{x∈C_c} d(x, μ_c)² (squared Euclidean distance). Random starts mean results vary, so use `n_init` (multiple runs, keep the best).
3. **K-means' three hidden assumptions:** (1) **round, equal-sized/equal-spread clusters**; (2) **hard membership** (each point in exactly one cluster); (3) **scale sensitivity** ("near" = Euclidean distance).
4. **Standardise first** (z = (x − x̄)/s). Otherwise large-scale features (income) dominate and the clustering becomes a sort on one column. Scaling silently decides which features get a vote.
5. **Choosing K is a claim you defend, not a number the data hands you.**
   - **Elbow:** plot inertia vs K. It **always falls**, so look for the **bend**. Real data rarely has a sharp corner.
   - **Silhouette:** s(i) = (b(i) − a(i)) / max(a(i), b(i)), ranging **−1 to +1**. a = mean distance to own cluster, b = mean distance to nearest other cluster. **+1 = well placed, ~0 = on the border, < 0 = probably misassigned.**
6. **On the credit data the two disagree:** the elbow hints at **3–4**, the silhouette peaks at **K = 2 (0.348)**. A silhouette of 0.35 is **weak/soft** structure. The elbow rewards **tightness** and the silhouette rewards **separation**. Print both, choose, and justify.
7. **Per-point silhouette ("knife") plot:** thin, ragged blades or bars below 0 are fence-sitters. **A healthy mean can hide a sick cluster.**
8. **Hierarchical (agglomerative) clustering:** start with every point alone and repeatedly **merge the two closest clusters** to build a **dendrogram**. **Cut the tree horizontally** to get clusters (your choice). Join height = how costly the merge was. The two knobs are the **distance** (between points) and the **linkage** (between clusters; **Ward** merges the pair that increases total inertia least). Weakness: **one bad early merge propagates**.
9. **Gaussian Mixture Model (GMM):** p(x) = Σ πₖ 𝒩(x | μₖ, Σₖ). Clusters are **tilted ellipses** with their own covariance, and assignment is **soft** (probabilities, e.g. 78%/22%). **K-means is the special case** with equal spherical covariances and probabilities rounded to 0/1. Choose K with **BIC**. The 50/50 points are the ones a human should review.
10. **Every clustering algorithm returns an answer on ANY data, even pure noise.** It can only tell you whether the data *tolerated* the structure you imposed.
11. **Internal validity** (silhouette, inertia) judges geometry only and needs no labels, but it is **blind to meaning, usefulness and fairness**. **External validity** compares to known labels, which you usually don't have. **The last word is a human**, not a metric.
12. **Curse of dimensionality:** as dimensions grow, points become roughly equidistant and the nearest/farthest neighbour gap shrinks, so distance-based methods weaken.
13. **PCA:** rotates the axes to the directions of **maximum variance**. PC1 = most variance; PC2 = next most, **orthogonal** to PC1. Mathematically these are the **eigenvectors of the covariance matrix** C = XᵀX/(n−1) (eigenvalue = variance captured). Projection: **Z = XVₖ**.
14. **Explained variance ratio** = λⱼ / Σλᵢ, shown on a **scree plot**. Credit data: **PC1 66%, PC2 20% → 86% in two axes** (PC3 8%, PC4 6%). Assumption: **variance = signal** (not always true, since rare low-variance directions like fraud get discarded first).
15. **Loadings name the axes.** PC1 loadings: income **+0.55**, credit history **+0.55**, debt ratio **−0.53**, prior defaults **−0.34**, which makes it a **"financial-health" axis**. An unnamed PC1/PC2 plot is unfinished. The **biplot** shows the loading arrows.
16. **PCA is the first LEARNED REPRESENTATION:** a new coordinate system fitted to the data. "A representation is a learned answer to *how should this data be described?*" Everything downstream runs on it.
17. **PCA is linear.** It fails on **curved structure** (two interleaved arcs overlap when projected onto PC1). "The variance was real; it just was not the signal."
18. **Embedding:** drop PCA's requirements that axes be straight, few and human-named. Each item becomes a point in a learned space where **position carries meaning**. **Distance = similarity** (nearest neighbour, recommendation, retrieval). **Direction = relationship** (**king − man + woman ≈ queen**, Mikolov et al. 2013).
19. **"The representation is the product."** Modern systems spend most effort learning the representation, and its quality **sets the ceiling** for everything above it.
20. **t-SNE / UMAP** give good **local** neighbourhoods, but **distances/gaps between clusters can be meaningless**. Don't read them as real distances.
21. **Representations encode bias:** recolour the PCA map by the (never supplied) protected group and the groups **separate along PC1**. **Group recoverable from PC1 alone: 0.711 vs base rate 0.426** (the supervised score in Weeks 3–4 leaked at ~0.70). "**Removing the column removed the word, not the information.**"
22. **Why:** an unequal world means correlated features, so the largest variance axis aligns with the inequality, and PCA (which seeks variance) finds it. It is **proxy leakage in geometric form**: "no malice, no bug, just correlation faithfully compressed".
23. **Nuance:** the hard K-means split matches the group only **weakly** (adjusted Rand index **0.156**, purity **0.701**). It is the **continuous PC1 axis** that carries the group strongly.
24. **A representation is never neutral.** Internal metrics measure **shape, not justice**: a clustering can be cohesive, well separated, high-scoring **and discriminatory**, and no silhouette will flag it.

---

## 1. What "unsupervised" means

| Supervised | Unsupervised |
|---|---|
| Inputs + correct answers | Inputs only |
| Reproduce the answer on unseen data | Propose a structure (groups, directions, lower-dim shape) |
| Accuracy, test set, stopping rule, overfitting check | None of that; no answer key |
| "Can I reproduce the known answer?" | "Is this proposed structure worth believing?" (a judgement) |

**Opening reveal:** take the credit applicants, remove the label **and** the group column, standardise four features, and cluster/project. The two groups still fall into two regions, because the group was hiding in the correlations.

---

## 2. K-means

**Algorithm (Lloyd's):**
1. **Assign** each point to its nearest centroid (Euclidean).
2. **Update** each centroid to the mean of its assigned points.
3. Repeat until assignments stop changing.

**Objective:** W = Σ_{c=1}^{K} Σ_{x∈C_c} d(x, μ_c)², where d(x,y) = √Σ(xⱼ − yⱼ)².

**Assumptions:** round, equal clusters; hard membership; scale sensitivity. When wrong, it "returns a confident, tidy answer that misreads the shape". If a clustering looks like a sort on one column, suspect the scaling.

---

## 3. Choosing K

| K | Inertia | Mean silhouette |
|---|---|---|
| 2 | 13,530 | **0.348** ← silhouette's choice |
| 3 | 10,630 | 0.311 |
| 4 | 8,638 | 0.265 |
| 8 | ~6,000 | lower |

- **Elbow:** inertia always decreases (K = n gives W = 0), so look for diminishing returns. A soft bend near 3.
- **Silhouette:** peaks at 2 and declines.
- Honest conclusion: "if I must choose, 2 is more defensible, but the structure is weak."
- Don't automate the choice of K. **The reasoning is the deliverable.**
- "If you're ever shown a perfect right-angle elbow, ask whether the data was simulated."

---

## 4. Hierarchical clustering and GMMs

### Comparison table

| | K-means | Hierarchical | GMM |
|---|---|---|---|
| Shape assumed | Equal round balls/spheres | Set by linkage | Tilted ellipses |
| Assignment | Hard | Hard, set by the cut | **Soft probabilities** |
| You must choose | K up front | Where to cut the tree | K and covariance form |
| Fails when | Clusters non-round or unequal | One bad early merge propagates | Wrong K invents ellipses |

- **Dendrogram:** low joins fuse similar points, tall joins fuse dissimilar groups. Cut low for many tight clusters, high for few loose ones.
- **GMM** has a likelihood, so it can use **BIC** (rewards fit, penalises parameters) for a more principled choice of K than the elbow.
- On two tilted, overlapping blobs: K-means slices with a straight line and mislabels the overlap, while GMM recovers the tilt and expresses the overlap as doubt.

---

## 5. Limits of cluster validity

| | Internal (silhouette, inertia) | External (vs known labels) |
|---|---|---|
| Needs labels? | No | Yes |
| Tells you | Tight & separated? | Do clusters match meaning? |
| Blind to | Meaning, usefulness, fairness, scaling artefacts | Usually unavailable |

"A clustering can be geometrically excellent and completely meaningless."

---

## 6. PCA

- **Curse of dimensionality:** everything becomes roughly equidistant, "nearest" loses meaning, and distance methods weaken. So compress first, then cluster or visualise.
- **Idea:** discard the spreadsheet's axes and find the directions of greatest variance. Orthogonal components, ordered by variance.
- **Maths:**
  - C = (1/(n−1)) XᵀX (covariance of the centred/standardised data)
  - C vⱼ = λⱼ vⱼ (eigenvectors = directions; eigenvalues = variance along them)
  - Z = X Vₖ (project onto the top k eigenvectors)
  - Explained variance ratio = λⱼ / Σλᵢ
- **Scree:** PC1 66.2%, PC2 19.7% (total **85.9%**), PC3 7.8%, PC4 6.3%. Reducing 4 → 2 costs ~14% of the variance.
- **Loadings:** income +0.549, credit history +0.547, debt ratio −0.533, prior defaults −0.339, giving a **financial-health / creditworthiness axis**.
- Coloured by K-means cluster, the split runs cleanly along PC1.

```python
Xs = StandardScaler().fit_transform(X)        # scale FIRST
pca = PCA().fit(Xs)
pca.explained_variance_ratio_                 # [0.66, 0.20, 0.08, 0.06]
Z = PCA(n_components=2).fit_transform(Xs)     # learned 2-D representation
pca.components_[0]                            # PC1 loadings
```

---

## 7. Representations and embeddings

- **Representation:** a learned description of the data. "The applicant did not change; their description did."
- **PCA's limit:** it is linear, so curved manifolds (interleaved arcs) collapse. This is why non-linear methods and learned embeddings exist.
- **Embedding:** a learned map of items (applicants, words, images, molecules) to points where **geometry = meaning**.
- "A representation is a space you have arranged so that the questions you care about become geometry."

| Method | Buys | Costs |
|---|---|---|
| PCA (linear) | Fast, interpretable axes, faithful **global** structure | Misses curved/non-linear structure |
| t-SNE / UMAP | Beautiful **local** neighbourhoods; clusters pop out | **Distances between clusters can be meaningless** |
| Learned embeddings | Task-tuned, richly expressive | Opaque axes, need data/compute, hard to audit |

---

## 8. Whose fingerprints are on the representation?

| Quantity | Value |
|---|---|
| Group recoverable from **PC1 alone** | **0.711** |
| Base rate (guess the majority group) | 0.426 |
| Week 4: group recoverable from supervised score | 0.70 |
| K-means (K=2) vs group: adjusted Rand index | 0.156 (weak) |
| K-means (K=2) vs group: purity | 0.701 |

**Causal chain:** unequal world → correlated features → variance aligned with inequality → PCA finds that axis → the axis recovers the group.

**Leak probe:**
```python
probe = LogisticRegression().fit(Z[:, [0]], sensitive)
probe.score(Z[:, [0]], sensitive)   # compare with base rate sensitive.mean()
```

"Whatever a dataset's protected attribute is, assume the representation can approximate it, and check. 'We removed that field' is a statement about the **schema**, not the **geometry**."

---

## 9. Lab and assessment

Four steps: **Cluster** (scale honestly) → **Validate** (elbow + silhouette, defend K) → **Reduce** (PCA, scree, 2-D, *name* PC1/PC2) → **Interrogate** (recolour by sensitive attribute; leak probe).

| Box-tick | Strong |
|---|---|
| "We found 3 clusters." | Names what each cluster is in the domain; which are shaky |
| "PCA explained 86% of variance." | Reads PC1 as a financial-health axis with loadings as evidence |
| Prints a PC1 vs PC2 scatter | Reports PC1 recovers the group at 0.71 and treats it as a finding |
| Ends there | States which structure it would refuse to act on, and why |

These moves return as **Elective 2** of the assignment.

### Five takeaways
1. Unsupervised learning proposes structure; it doesn't discover truth.
2. Choosing K, the cut or the method is a modelling decision with your name on it.
3. PCA is your first learned representation, not just compression.
4. Geometry as meaning is the idea the whole course rests on.
5. A representation is never neutral; the last word is a human, not a silhouette.

**Key readings:** Jolliffe & Cadima (2016) PCA review; Rousseeuw (1987) silhouettes; van der Maaten & Hinton (2008) t-SNE; McInnes et al. (2018) UMAP; Mikolov et al. (2013) word vectors; Wattenberg et al. (2016) *How to Use t-SNE Effectively*; Bishop ch. 9 (GMM/EM).

---

## ✅ Quick self-check (MCQ style)

1. A silhouette value of −0.2 for a point means: **(a)** it is well clustered **(b)** it sits exactly on the border **(c)** it is closer to a neighbouring cluster than its own (likely misassigned) **(d)** it is an outlier removed by K-means. → **c**
2. Which is TRUE of K-means compared with a GMM? **(a)** K-means gives soft probabilities **(b)** K-means is the special case of a GMM with equal spherical covariances and hard assignments **(c)** a GMM requires no choice of K **(d)** K-means handles tilted ellipses better. → **b**
3. Within-cluster inertia as K increases: **(a)** always decreases **(b)** always increases **(c)** is minimised at the true K **(d)** is unaffected by scaling. → **a**
4. The principal components of standardised data are: **(a)** cluster centroids **(b)** eigenvectors of the covariance matrix **(c)** the original features ranked by variance **(d)** the loadings of a GMM. → **b**
5. In t-SNE/UMAP plots, which is the most common misreading? **(a)** treating local neighbours as similar **(b)** treating gaps between clusters as meaningful distances **(c)** using colour to show labels **(d)** using 2 dimensions. → **b**
