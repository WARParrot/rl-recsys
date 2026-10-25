# Superseded Pilot 1 — historical appendix

> This file is retained for provenance and historical detail. Its old pilot tables are **not** the final summary and must not be mixed with the prospective Experiment A contract or completion-only follow-up. The core assignment result is the executed joint-RM pilot described here; the full reviewer archive is available in the [Google Drive folder](https://drive.google.com/drive/folders/12vRYPFrcBkTOqQ7nwDjYAFQzk78cNk2k?usp=sharing).

> **Основной актуальный отчёт — [README.md](README.md).** Разделы 1–8 ниже описывают старый запуск и сохраняются для истории. Раздел 9 содержит последующее объяснение нового запуска; актуальные выводы, источники, метрики и инструкции объединены в README. Старые численные таблицы и замечания о source-byte identity не относятся к релизу v0.2.0-pilot-2.

**Статус:** historical joint-RM restricted pilot; это не доказательство causal/user impact. Проспективный research-suite реализован, но его завершённый результат здесь не заявляется. Completion-only follow-up выполнен отдельно и описан в [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).

**Provenance qualification:** metadata hashes agree internally, but the supplied run's source-code hashes could not be reconstructed from the recorded Git revision/current checkout, including line-ending checks. Exact source-byte identity remains unverified. Do not treat the revision label alone as proof of the executed source.

## 1. Вопрос и scope

Проверялось, может ли дообучение последовательной item-ID модели относительно learned reward model увеличить completion proxy без practically meaningful потери next-item relevance.

Политика: GPT-2-style decoder, обученный с нуля на item IDs. RL: simulator-on-policy REINFORCE относительно детерминированного append-item simulator, с KL к frozen SFT policy. Это **model-based policy optimization**, а не on-policy обучение относительно реальных пользователей: сгенерированный item не становится наблюдаемым пользовательским действием.

Основной pilot: Yambda-50M `flat/50m/listens.parquet`, 1,000 детерминированно выбранных пользователей, train-known каталог 20,000 items, context length 64, seed 42.

## 2. Evidence ledger

| Claim / assumption | Source | Establishes | Limits / local validation |
|---|---|---|---|
| Yambda содержит listen feedback, timestamp, organic flag, playback ratio и track duration | [Yambda dataset card](https://huggingface.co/datasets/yandex/yambda), revision `dd6f3a19eef5866e346c3270e098baa641a44948` | Схему и масштабы dataset | Dataset card не делает observed listen causal exposure label; локально SHA-256 проверен |
| Timestamp bins не задают порядок событий внутри bin | Yambda dataset card; local `audit.json` | Timestamps bin-ованы по 5 секунд | Локальный протокол исключает ambiguous user-timestamp groups |
| `played_ratio_pct > 100` возможен из-за replay | Yambda FAQ | Это не обязательно data error | Reward bounded/capped; proxy не равен wall-clock listening |
| Generative recommendation can be aligned with secondary goals using RL | Petrov & Macdonald, *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning*, [arXiv:2403.04875](https://arxiv.org/abs/2403.04875) | Близкий methodological precedent | Их data/task/candidate setup не переносится автоматически на этот pilot |
| Learned feedback model and multi-step off-policy/model-based evaluation have limitations | Matveev et al., *Session-Level Optimization...*, [arXiv:2607.02818](https://arxiv.org/abs/2607.02818) | Близкий session-level formulation | Не causal validation of this implementation; paper uses a different retriever/protocol |
| Local model-based reward does not identify user satisfaction | Methodological audit: `audits/methodology.md`, `audits/code_methodology.md` | Observational/exposure and simulator limitations | No propensity logs or randomized online experiment available |

## 3. Declared protocol and provenance

- Dataset SHA-256: `eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1`
- Training/evaluation code revision: `7892d187be26546cad71ff2d8db4d6b216f96f33` (the release also includes this updated report).
- Run artifacts: [GitHub Release v0.1.0-pilot](https://github.com/WARParrot/rl-recsys/releases/tag/v0.1.0-pilot). The archive includes audit, provenance, selection, all candidates, results, training/progress logs and `READY_FOR_TEST.json`, plus SHA-256 checksums. Checkpoints and prepared data are not included.
- Environment: Windows 10, Python 3.11.9, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5060 Ti 16 GB.
- Temporal role cutoffs: `[19853890, 22072765, 24049600]`, yielding fit/calibration/selection/test roles (70/10/10/10 by timestamp quantile declaration).
- Exact duplicates removed: 22,634 of 5,251,122 input rows.
- Ambiguous timestamp rows excluded: 1,446,115.
- `played_ratio_pct > 100`: 19,463 rows.
- Retained caps: fit 250,000; calibration 10,000; selection 10,000; test 10,000.
- Test target coverage in final all-test policy evaluation: 0.5083; reward-model test after eligible filtering: 5,083 examples / 733 users.

This report supersedes the earlier pilot numbers with the rerun on `7892d18`. All five metadata hashes in `READY_FOR_TEST.json` match the supplied selection, candidates, training, provenance and audit files. Its selected checkpoint agrees with selection; hashes for all 15 checkpoints are recorded, but the checkpoint bytes were not supplied for independent verification. This establishes internal artifact consistency, not an independently timestamped preregistration.

The Windows evaluation stderr still contains a PyTorch redirect warning rendered as a PowerShell NativeCommandError. It is now separate from the results JSON; no Python traceback was found. A process exit-status manifest was not supplied, so a clean wrapper exit is not independently established. The provenance lists only untracked logs/egg-info, not modifications to tracked source files.

The historical selector maximized all-context sampled reward while the paired test reward comparison used eligible-target contexts. This population mismatch does not change the negative practical conclusion, but prevents treating selection and test reward as exactly the same estimand. The revised code aligns both to eligible contexts. The historical seconds head was optimized but not held-out validated separately; its duration-shift interpretation is exploratory.

All-target policy metrics average over 10,000 contexts / 837 users. Eligible relevance averages over 5,083 targets / 733 users. Selection uses eligible NDCG; paired sampled-reward comparisons also use eligible contexts and must not be confused with the all-target sampled-reward column below.

## 4. Hypotheses and decision rules

- **H1:** history-aware reward model improves test user-macro MAE over item-mean by at least 2%; supported only if the paired user-bootstrap relative 95% interval is below -2% and the user count is adequate.
- **H2:** selected completion-RL policy improves sampled completion proxy by at least 5% relative to SFT while eligible NDCG remains above 98% of SFT. Both conditions required. This is proxy-only and not causal.
- **H3:** five-step synthetic rollouts show increasing model/context shift relative to one-step. Exploratory only; no confirmatory threshold was declared.

Bootstrap intervals are conditional on seed 42. They do not quantify training-seed uncertainty.

## 5. Results

### Reward model test

| Predictor | User-macro MAE |
|---|---:|
| Constant | 0.35698 |
| Train item mean | 0.34776 |
| History-aware reward model | 0.33976 |
| Mean-history architectural auditor | 0.35202 |

History-aware point estimate vs item mean: `-2.30%`; paired user-bootstrap relative 95% CI: `[-5.32%, +0.87%]`. Calibration is poor: low predictions underpredict observed completion and high predictions overpredict it. Organic slice MAE is 0.37577 vs 0.31823 for recommendation-driven events. The auditor has independent parameters and a different architecture, but shares the fit/calibration data and observational bias; it is not independent ground truth.

### Policy test

| Policy | Recall@10 | NDCG@10 | Eligible NDCG@10 | Sampled reward | Auditor reward | Repeat rate | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00829 | 0.00326 | 0.00631 | 0.72284 | 0.72271 | 0.03855 | 0.00050 |
| SFT | 0.03697 | 0.02156 | 0.04461 | 0.72327 | 0.72195 | 0.12105 | 0.29545 |
| Reward rerank | 0.03687 | 0.02177 | 0.04513 | 0.72393 | 0.72190 | 0.12089 | 0.29515 |
| RL completion H=1, beta=0.1 | 0.03661 | 0.02197 | 0.04542 | 0.72385 | 0.72158 | 0.12129 | 0.29210 |
| RL completion H=5, beta=0.1 | 0.03542 | 0.02215 | 0.04529 | 0.72338 | 0.72208 | 0.11835 | 0.29545 |
| RL seconds H=1, beta=0.01 | 0.03511 | 0.02225 | 0.04540 | 0.72367 | 0.72184 | 0.11682 | 0.28885 |

The selection-only chosen candidate was `rl_completion_h1_b0.1`. On eligible test contexts, sampled reward changed from 0.729108 to 0.729444: `+0.046%`, paired user-bootstrap relative CI `[+0.005%, +0.091%]`, far below 5%. Eligible NDCG changed by `+1.82%`, CI `[-0.64%, +4.88%]`: noninferiority at the declared -2% margin is supported conditionally on this seed, but superiority is not. All-target Recall changed by approximately `-0.98%` (point estimate); it uses a different population from eligible NDCG. All other grid comparisons are exploratory, not post-test model selection.

### Synthetic five-step diagnostic

For SFT, reward-model predicted reward rose from 0.72319 at step 1 to 0.75024 at step 5, while auditor reward changed from 0.72497 to 0.72268. For selected RL H=1 beta=0.1, reward-model reward rose from 0.72345 to 0.75019, while auditor reward changed from 0.72558 to 0.72273. This is a divergence in mean-score trajectories, not evidence of user satisfaction or long-term value. Mean absolute per-example disagreement actually declined (selected RL: 0.13217 to 0.12098), so H3 should not be described as measured monotonic growth in disagreement.

### Full RL grid and behavioral trade-offs (exploratory)

All 12 candidates are retained, including unfavorable outcomes. Completion reward and seconds reward are distinct bounded heads; the seconds column is normalized by a 600-second cap, not measured wall-clock time. Columns below use all-target contexts except eligible NDCG.

| Reward | H | beta | Eligible NDCG | Completion proxy | Seconds proxy | Coverage | HHI | Mean top-10 duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 1 | 0.01 | 0.04386 | 0.72390 | 0.22737 | 0.28680 | 0.00119 | 192.63 |
| completion | 1 | 0.1 | 0.04542 | 0.72385 | 0.22740 | 0.29210 | 0.00107 | 193.60 |
| completion | 1 | 1 | 0.04531 | 0.72335 | 0.22742 | 0.29465 | 0.00103 | 194.47 |
| completion | 5 | 0.01 | 0.04140 | 0.72315 | 0.22727 | 0.29410 | 0.00142 | 192.25 |
| completion | 5 | 0.1 | 0.04529 | 0.72338 | 0.22732 | 0.29545 | 0.00106 | 193.84 |
| completion | 5 | 1 | 0.04521 | 0.72337 | 0.22735 | 0.29570 | 0.00103 | 194.40 |
| seconds | 1 | 0.01 | 0.04540 | 0.72367 | 0.22782 | 0.28885 | 0.00110 | 198.77 |
| seconds | 1 | 0.1 | 0.04493 | 0.72353 | 0.22750 | 0.29375 | 0.00104 | 195.79 |
| seconds | 1 | 1 | 0.04472 | 0.72333 | 0.22747 | 0.29510 | 0.00103 | 194.96 |
| seconds | 5 | 0.01 | 0.04382 | 0.72317 | 0.22784 | 0.28540 | 0.00133 | 199.58 |
| seconds | 5 | 0.1 | 0.04517 | 0.72321 | 0.22743 | 0.29545 | 0.00101 | 196.05 |
| seconds | 5 | 1 | 0.04471 | 0.72331 | 0.22740 | 0.29495 | 0.00103 | 194.89 |

- **KL:** beta=1 keeps duration, concentration and coverage close to SFT. Smaller beta allows more drift, but is not uniformly better: completion H=5/beta=0.01 has eligible NDCG 0.04140 versus SFT 0.04461, with HHI increasing to 0.00142.
- **Reward definition:** seconds H=1/beta=0.01 shifts mean recommended duration to 198.77 s versus SFT 194.68 s; completion H=1/beta=0.01 shifts it to 192.63 s. This is consistent with duration shortcuts, not proof of their causal mechanism.
- **Concentration:** selected RL coverage is 29.21% versus SFT 29.55%; HHI is approximately 0.00107 versus 0.00103, far below popularity's 0.1. No aggregate collapse to the same ten items is observed, but catalog coverage alone does not establish personalization or within-list diversity.
- **Trade-off:** reranking has a higher all-target completion proxy than every RL candidate in this run (0.72393) while retaining eligible NDCG 0.04513. Selected RL has slightly higher eligible NDCG, so neither dominates the other on both axes. There is no compelling practical RL advantage. These are point-estimate comparisons, not multiplicity-adjusted superiority tests.

## 6. Verdicts

### H1 — **Not supported**

The point estimate crosses the 2% materiality threshold, but the uncertainty interval includes zero and positive degradation. The reward model is not validated strongly enough to serve as a reliable optimization target.

### H2 — **Not supported**

The selected RL policy increased the sampled proxy by about 0.046% on eligible contexts, not 5%. Reranking achieved a comparable proxy/relevance trade-off without policy training. The result does not support rollout or product claims.

### H3 — **Inconclusive for increasing disagreement; exploratory score divergence**

Five-step synthetic contexts produce higher reward-model scores but not higher auditor scores, while mean absolute disagreement decreases. This does not support increasing disagreement. Model exploitation/context shift remains a hypothesis, not a directly measured mechanism; the auditor shares observational exposure bias and cannot establish causal reward hacking.

## 7. Limitations and validity threats

1. Logged listens are conditional on exposure and prior policy; no propensities, randomized treatment, or causal estimand is available.
2. Reward model, simulator and optimization target form a potentially exploitable loop. The auditor is architecturally different but not independent of observational bias.
3. Only one training seed was run; bootstrap intervals are not seed robustness intervals.
4. The pilot excludes ambiguous timestamp rows and restricts the catalog to train-known top 20,000 items, so results do not generalize to all Yambda items/users.
5. A sampled next-listen target is not a complete relevance label for unseen unexposed items.
6. `played_ratio_pct * duration` is only a capped proxy; it is not exact wall-clock listening.
7. Windows stderr contains a redirect warning; structured results were generated, but no independent process exit-status manifest is available.
8. One sampled action per context is used. The paired user bootstrap is conditional on that Monte Carlo draw and checkpoint, not a complete estimate of sampling/training uncertainty.
9. The test has been inspected in earlier pilots; this rerun is not a fresh untouched confirmatory dataset. Its intervals are descriptive, and a materially revised hypothesis needs a new frozen evaluation period.

## 8. Recommendation and remaining historical context

Do not claim that RL improved recommendations or user satisfaction. Treat this pilot as a negative/diagnostic result: under this reward, simulator and KL settings, model-based RL did not produce a practically meaningful gain over SFT or reranking.

The historical pilot recommended the following directions before the later completion-only follow-up; they are retained as context, not open requirements for this submission:

1. retain separated Windows stdout/stderr and add process exit/status manifests;
2. use a strict independent reward-audit split and quantify OOD coverage of recommended actions;
3. add exposure-aware logged evaluation if propensities/candidate impressions become available;
4. compare RL and reward reranking under the same reward budget and candidate universe;
5. add explicit context-drift and reward-model disagreement diagnostics for H=5;
6. run additional seeds only with a predeclared success criterion.

The completion-only follow-up has now supplied the reward-variant result described in [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md). The remaining items are prospective extensions, not blockers for documenting the historical pilot.
### Long-term objectives: proposed, not measured in this pilot

A future reward could combine immediate completion with a separate history-conditioned continuation/return head: e.g. another listen within a declared session gap and return during a fixed 7-day horizon, with penalties for repetition/concentration. Timestamp units and the dataset's observation horizon must first be validated. Labels need an adequate future observation window; right-censored examples must be excluded or modeled explicitly, and temporal training/evaluation boundaries must be purged by the label horizon to prevent leakage. No such retention head was trained here.

Offline experiments can assess time-forward predictive quality/calibration of these labels and reward/relevance sensitivity. They cannot establish that a recommendation caused a return: user intent, exposure and the logging policy confound that association. A randomized user-level online test with retention, satisfaction, relevance/diversity guardrails and a predeclared observation period is needed for product conclusions. Appending generated items is not a substitute for observed acceptance or return behavior.

The current decision is **not to roll out** and **not to spend additional compute reproducing the same pilot**.

## 9. Evidence-grounded interpretation addendum

This addendum uses the machine-readable run summarized in `RESULTS.md` (`rl-pilot2` / `pilot-seed42`), and must not be mixed with the older table above: the runs select different RL candidates and use different primary reward summaries. The uploaded run is explicitly `exploratory_reuse`, not a fresh confirmatory test.

### Verified primary references and limits

1. **Williams (1992), “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,” *Machine Learning*.** DOI: [10.1007/BF00992696](https://doi.org/10.1007/BF00992696). Establishes the likelihood-ratio policy-gradient estimator and variance-reducing baselines under its trajectory assumptions. **Limit:** it validates the estimator for the implemented synthetic transition process, not the learned reward, user causality, or training-seed robustness.
2. **Schulman et al. (2015), “Trust Region Policy Optimization.”** [arXiv:1502.05477](https://arxiv.org/abs/1502.05477). Derives a monotonic-improvement bound for a trust-region surrogate and then approximates it in practical TRPO. **Limit:** this implementation uses Adam plus a per-state forward KL penalty; it is not TRPO and inherits no monotonic-improvement guarantee.
3. **Janner, Fu, Zhang, and Levine (2019), “When to Trust Your Model: Model-Based Policy Optimization.”** [arXiv:1906.08253](https://arxiv.org/abs/1906.08253). Analyzes model-generated-data bias and reports that short rollouts branched from real data can be safer than long model rollouts. **Limit:** it does not validate this append-item simulator; the H=5 states here contain generated items, not observed user acceptance or feedback.
4. **Talvitie (2017), “Self-Correcting Models for Model-Based Reinforcement Learning.”** [arXiv:1612.06018](https://arxiv.org/abs/1612.06018). Shows that small one-step model errors can compound during planning and that multi-step performance depends on self-correction, not only one-step prediction error. **Limit:** no self-correcting user-transition model is trained here.
5. **Skalse, Howe, Krasheninnikov, and Krueger (2022), “Defining and Characterizing Reward Hacking.”** [arXiv:2209.13085](https://arxiv.org/abs/2209.13085). Defines reward hacking as improving an imperfect proxy while worsening the true reward and characterizes when proxies can be unhackable. **Limit:** the paper does not show that this run hacked reward; it supports treating proxy/auditor divergence as a warning, not as proof.
6. **Gao, Schulman, and Hilton (2022), “Scaling Laws for Reward Model Overoptimization.”** [arXiv:2210.10760](https://arxiv.org/abs/2210.10760). In a synthetic setup with a known gold reward, reports that sufficiently optimizing a proxy can reduce the gold score and studies KL effects. **Limit:** this pilot has no gold user-satisfaction signal, so the result motivates diagnostics but cannot identify a true-reward decline here.
7. **Jiang and Li (2016), “Doubly Robust Off-policy Value Evaluation for Reinforcement Learning,” ICML 33, 652–661.** [PMLR](https://proceedings.mlr.press/v48/jiang16.html). Establishes assumptions and bias/variance trade-offs for sequential off-policy evaluation. **Limit:** this project has neither logged propensities nor randomized exposure and does not perform OPE; next-listen metrics are predictive/associational.

### Report-ready causal chain

**What was optimized.** At each step the code samples an item from the current policy, evaluates the frozen observational reward model on the pre-action context and that item, then appends the item to form the next synthetic context. REINFORCE optimizes these scores with an exact categorical KL penalty to frozen SFT. For H=1, exact evaluation integrates the policy distribution over the restricted 20,000-item catalog. For H=5, the transition is repeated appending; no acceptance, rejection, or user response is sampled. The computational chain is therefore *parameters → action distribution → frozen-RM score*, with *sampled action → appended context → subsequent scores* at later steps. This identifies a model-score objective, not satisfaction, retention, or real-user policy value.

**Observed result.** In the uploaded run, the selected `rl_completion_h1_b0.01` changed eligible exact one-step expected frozen-RM score from 0.729139 (SFT) to 0.729696: +0.0763%, with paired user-bootstrap relative interval [+0.0705%, +0.0823%]. Eligible NDCG@10 changed from 0.044494 to 0.044024 (−1.06%; interval approximately [−4.50%, +2.61%]). These comparisons use 5,083 eligible contexts / 733 users; intervals are conditional on this trained seed, not seed robustness. H2 required at least +5% reward with NDCG non-inferiority at −2%; neither practical reward gain nor relevance non-inferiority is established. Reranking reached 0.729961 eligible expected score and 0.045192 eligible NDCG, beating selected RL on both point estimates, without a paired superiority test being asserted. These are reused-test descriptive results, not causal estimates. The all-target expected scores (0.722888, 0.723430, and 0.723723 respectively) describe a different population and must not be substituted into the eligible paired comparison.

**KL and horizon.** The grid is consistent with a drift–trade-off: weak KL plus H=5 produced the clearest relevance loss (eligible NDCG 0.041887 versus 0.044494 for SFT, about −5.86%) without a material proxy gain. This is compatible with KL limiting policy drift and synthetic state/model error accumulating, but it is not a controlled causal attribution: only one seed was run and beta, horizon, and optimization dynamics are jointly varied. KL is a reference-policy regularizer, not a reward-validity or safety certificate.

**Synthetic rollouts: model-error hypothesis, not measured hacking.** SFT RM score rose from 0.722641 at synthetic step 1 to 0.750423 at step 5, while the architecture-different auditor was approximately flat (0.723147 to 0.722932). For selected H=1 RL, RM rose from 0.723082 to 0.750449 while auditor changed from 0.723190 to 0.722625. However, RM–auditor mean absolute disagreement declined (approximately 0.13235 to 0.12259), so monotonic disagreement growth was not measured. The supported statement is: *the optimized RM assigns increasing scores on generated contexts while a second observational model does not show the same increase.* This is compatible with reward-model extrapolation, shortcut exploitation, or simulator/state-distribution shift. It is not evidence that true user reward decreased, because no gold reward or user transition is observed; therefore “reward hacking” must remain a hypothesis, not a finding. The sampled-action support diagnostic (roughly 0.4% unsupported actions for selected RL) is an OOD warning, not a propensity estimate.

**Conclusion.** The strongest defensible conclusion is negative and conditional: in this restricted catalog, one seed, frozen observational reward, deterministic append-item simulator, and tested KL/horizon grid, RL did not reach the predeclared practically meaningful proxy improvement, and reranking was a stronger simple comparator. The evidence is compatible with weak reward modeling, simulator/state-distribution shift, and KL-constrained limited movement, but cannot separate those mechanisms. It does not support user-impact, satisfaction, retention, causal-value, or “on-policy user RL” claims. A follow-up should prioritize an independent reward audit, matched rollout controls, and seed robustness. A later untouched period would strengthen generalization evidence. Real-user policy-value claims additionally need a defensible identification strategy: for example, randomized exposure, propensity-based OPE with support assumptions, or a sufficiently supported direct outcome/transition model under explicit assumptions. Missing propensities alone do not rule out every direct-model approach; this pilot has not established such a model.
