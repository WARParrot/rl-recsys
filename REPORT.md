# Yambda-50M: GPT-2 item-ID + model-based RL pilot

**Статус:** завершён один restricted pilot; это не доказательство causal/user impact.

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
- Code revision: `13f099cbb7debf6bcf42c35177f3f1a80d4f757c`
- Environment: Windows 10, Python 3.11.9, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5060 Ti 16 GB.
- Temporal role cutoffs: `[19853890, 22072765, 24049600]`, yielding fit/calibration/selection/test roles (70/10/10/10 by timestamp quantile declaration).
- Exact duplicates removed: 22,634 of 5,251,122 input rows.
- Ambiguous timestamp rows excluded: 1,446,115.
- `played_ratio_pct > 100`: 19,463 rows.
- Retained caps: fit 250,000; calibration 10,000; selection 10,000; test 10,000.
- Test target coverage in final all-test policy evaluation: 0.5083; reward-model test after eligible filtering: 5,083 examples / 733 users.

The supplied `evaluation.log` contains a Windows PowerShell/`torch.distributed.elastic` redirect warning. `results.json` was produced and contains the complete policy table, so numerical artifacts are analyzable; the wrapper exit status should be treated as not clean and fixed before a formal rerun.

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
| History-aware reward model | 0.33974 |
| Independent mean-history auditor | 0.35202 |

History-aware point estimate vs item mean: `-2.30%`; paired user-bootstrap relative 95% CI: `[-5.35%, +0.85%]`. Calibration is poor: low predictions underpredict observed completion and high predictions overpredict it. Organic slice MAE is 0.37576 vs 0.31819 for recommendation-driven events.

### Policy test

| Policy | Recall@10 | NDCG@10 | Eligible NDCG@10 | Sampled reward | Auditor reward | Repeat rate | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.00829 | 0.00326 | 0.00631 | 0.72281 | 0.72271 | 0.03855 | 0.00050 |
| SFT | 0.03697 | 0.02153 | 0.04449 | 0.72325 | 0.72196 | 0.12093 | 0.29555 |
| Reward rerank | 0.03687 | 0.02177 | 0.04514 | 0.72389 | 0.72196 | 0.12085 | 0.29525 |
| RL completion H=1, beta=0.1 | 0.03661 | 0.02197 | 0.04543 | 0.72382 | 0.72160 | 0.12133 | 0.29165 |
| RL completion H=5, beta=0.1 | 0.03554 | 0.02222 | 0.04552 | 0.72327 | 0.72194 | 0.11807 | 0.29520 |
| RL seconds H=1, beta=0.01 | 0.03533 | 0.02196 | 0.04515 | 0.72358 | 0.72178 | 0.11642 | 0.28960 |

The selected candidate was `rl_completion_h1_b0.1`. On test, its sampled reward relative difference vs SFT was `+0.042%`, with bootstrap relative CI `[0.000%, +0.088%]`; far below the declared 5% threshold. Eligible NDCG relative difference was `+2.10%`, CI `[-0.37%, +5.16%]`; Recall relative difference was `-0.79%`, CI `[-2.98%, +1.26%]`.

### Synthetic five-step diagnostic

For SFT, reward-model predicted reward rose from 0.72317 at step 1 to 0.75078 at step 5, while auditor reward stayed approximately flat (0.72497 to 0.72269). For selected RL H=1 beta=0.1, reward-model predicted reward rose from 0.72349 to 0.75088, while auditor reward was 0.72561 at step 1 and 0.72279 at step 5. This is evidence of simulator/model disagreement, not evidence of user satisfaction or long-term value.

## 6. Verdicts

### H1 — **Not supported**

The point estimate crosses the 2% materiality threshold, but the uncertainty interval includes zero and positive degradation. The reward model is not validated strongly enough to serve as a reliable optimization target.

### H2 — **Not supported**

The selected RL policy increased the sampled proxy by about 0.04%, not 5%. Reranking achieved a comparable proxy/relevance trade-off without policy training. The result does not support rollout or product claims.

### H3 — **Exploratory evidence only**

Five-step synthetic contexts produce higher reward-model scores but not higher auditor scores. This is consistent with model exploitation/context shift, but the auditor shares observational exposure bias and cannot establish causal reward hacking.

## 7. Limitations and validity threats

1. Logged listens are conditional on exposure and prior policy; no propensities, randomized treatment, or causal estimand is available.
2. Reward model, simulator and optimization target form a potentially exploitable loop. The auditor is architecturally different but not independent of observational bias.
3. Only one training seed was run; bootstrap intervals are not seed robustness intervals.
4. The pilot excludes ambiguous timestamp rows and restricts the catalog to train-known top 20,000 items, so results do not generalize to all Yambda items/users.
5. A sampled next-listen target is not a complete relevance label for unseen unexposed items.
6. `played_ratio_pct * duration` is only a capped proxy; it is not exact wall-clock listening.
7. The supplied evaluation wrapper emitted a Windows redirect warning; artifacts were generated, but wrapper correctness should be repaired before a confirmatory rerun.

## 8. Recommendation and next experiment

Do not claim that RL improved recommendations or user satisfaction. Treat this pilot as a negative/diagnostic result: under this reward, simulator and KL settings, model-based RL did not produce a practically meaningful gain over SFT or reranking.

The next experiment should be materially different rather than another seed of the same setup:

1. fix the Windows evaluation wrapper and add a clean exit/status manifest;
2. use a strict independent reward-audit split and quantify OOD coverage of recommended actions;
3. add exposure-aware logged evaluation if propensities/candidate impressions are available, or explicitly restrict claims to conditional predictive ranking;
4. compare RL and reward reranking under the same reward budget and candidate universe;
5. add explicit context-drift and reward-model disagreement penalties for H=5;
6. run additional seeds only after the revised protocol has a predeclared success criterion.

The current decision is **not to roll out** and **not to spend additional compute reproducing the same pilot**.
