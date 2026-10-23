# Yambda-50M: GPT-2 item-ID + model-based RL pilot

**Статус:** исторический restricted pilot; это не доказательство causal/user impact. Текущий код обновлён после аудита: [protocol amendment](RESEARCH_UPGRADE.md). Нового GPU-эксперимента по обновлённому протоколу пока нет; таблицы ниже не пересчитаны.

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

## 8. Recommendation and next experiment

Do not claim that RL improved recommendations or user satisfaction. Treat this pilot as a negative/diagnostic result: under this reward, simulator and KL settings, model-based RL did not produce a practically meaningful gain over SFT or reranking.

The next experiment should be materially different rather than another seed of the same setup:

1. retain the separated Windows stdout/stderr and add a process exit/status manifest;
2. use a strict independent reward-audit split and quantify OOD coverage of recommended actions;
3. add exposure-aware logged evaluation if propensities/candidate impressions are available, or explicitly restrict claims to conditional predictive ranking;
4. compare RL and reward reranking under the same reward budget and candidate universe;
5. add explicit context-drift and reward-model disagreement penalties for H=5;
6. run additional seeds only after the revised protocol has a predeclared success criterion.

### Long-term objectives: proposed, not measured in this pilot

A future reward could combine immediate completion with a separate history-conditioned continuation/return head: e.g. another listen within a declared session gap and return during a fixed 7-day horizon, with penalties for repetition/concentration. Timestamp units and the dataset's observation horizon must first be validated. Labels need an adequate future observation window; right-censored examples must be excluded or modeled explicitly, and temporal training/evaluation boundaries must be purged by the label horizon to prevent leakage. No such retention head was trained here.

Offline experiments can assess time-forward predictive quality/calibration of these labels and reward/relevance sensitivity. They cannot establish that a recommendation caused a return: user intent, exposure and the logging policy confound that association. A randomized user-level online test with retention, satisfaction, relevance/diversity guardrails and a predeclared observation period is needed for product conclusions. Appending generated items is not a substitute for observed acceptance or return behavior.

The current decision is **not to roll out** and **not to spend additional compute reproducing the same pilot**.
