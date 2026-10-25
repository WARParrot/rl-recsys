# Методологический аудит simulator-on-policy/model-based RL-рекомендаций

**Статус:** read-only аудит постановки; код не аудировался и не изменялся.  
**План:** [`PLAN.md`](../docs/archive/PLAN.md).
**Терминология:** далее используется `simulator-on-policy/model-based policy optimization`: траектории on-policy относительно learned simulator/reward model, но **не** относительно реальных пользователей и не causal offline policy evaluation.

Аудированы постановка из `Pasted-text.txt`, планы проекта и источники:

- Petrov & Macdonald, *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning*, arXiv:2403.04875;
- Matveev et al., *Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction*, arXiv:2607.02818v2 (статус и датировка версии требуют отдельной проверки);
- dataset card/описание Yambda-5B.

## 1. Итоговый вердикт

План в `PLAN.md` после уточнения терминов является методологически приемлемым как ограниченный model-based pilot, если сохраняются следующие границы:

1. Основной RL estimand — ожидаемый bounded completion reward **замороженной reward model**, а не удовлетворённость пользователя.
2. Relevance — отдельный predictive offline estimand по следующему logged listen, не causal effect.
3. Все сгенерированные items на горизонте 5 — гипотетический context, а не доказательство фактического пользовательского принятия.
4. RM validation на observed listens не считается валидацией на unexposed catalog actions.
5. Full eligible catalog, train-only vocabulary, временные границы и repeat policy фиксируются одинаково для всех методов.
6. Checkpoint selection выполняется только на selection interval; test остаётся frozen.
7. Любой результат без независимого RM audit и без sensitivity analysis трактуется только как model-score optimization.

Ключевой риск уже явно смягчён планом: не заявлять on-policy относительно пользователей. Остаются ограничения simulator validity, observational labels, support/coverage и multi-step state shift.

## 2. Estimands и что именно можно утверждать

### 2.1 One-step relevance

Для test context `s` и observed next listen `a` основной predictive estimand:

\[
\operatorname{NDCG@K}(\pi)=
\mathbb E_{(s,a)\sim D_{test}}
[\operatorname{NDCG@K}(\pi,s,a)].
\]

Это условная способность ранжировать observed target относительно заранее определённого eligible candidate set. Это **не** вероятность эффекта вмешательства `do(A=a)` и не оценка качества всех unseen items.

План правильно требует:

- user-macro aggregation;
- all-target и eligible-target views;
- excluded/unknown target counts;
- строго более раннюю историю;
- отдельную отчётность target coverage.

Нужно явно фиксировать, что unknown/excluded target получает zero только в all-target view; в eligible-target view denominator и population должны быть объявлены до просмотра финальных результатов.

### 2.2 Model-based optimization

Если `\hat r` — frozen RM и `\hat P` — learned transition/simulator, то оптимизируемый estimand:

\[
\hat V_{\hat P,\hat r}(\pi)=
\mathbb E_{\tau\sim \pi,\hat P}
\left[\sum_{t=0}^{H-1}\gamma^t\hat r(s_t,a_t)\right].
\]

Корректное название — simulator/model-based value. Нельзя называть его observed user satisfaction или causal policy value. Для `H=5` каждый следующий state должен быть явно определён: append item, predicted feedback, termination, repeat handling и organic events. Если state просто получает item без sampled/predicted feedback, это open-loop autoregressive rollout, а не полноценная user simulator trajectory.

### 2.3 Causal limitation

В logged data наблюдается outcome только для фактически exposed action. В общем случае `\hat r(s,a)` обучает ассоциацию в области logging policy, а не:

\[
\mathbb E[Y\mid do(A=a),S=s].
\]

Для causal claim необходимы treatment definition, time zero, pre-treatment covariates, exposure/propensity, positivity, temporal ordering, interference assumptions и randomized online test либо обоснованный OPE. План правильно не использует IPS/DR claims; это ограничение необходимо сохранить в отчёте.

## 3. Блокирующие/критические угрозы валидности

### 3.1 RM как objective и источник circularity

Policy максимизирует RM score. Поэтому рост RM reward может быть:

- reward hacking;
- shortcut по популярности, duration или history length;
- повтором already-consumed items;
- extrapolation в OOD области;
- эксплуатацией ошибок calibration;
- artefact-ом item ID или padding/masking.

Независимый RM auditor обязателен. Он должен быть обучен/настроен так, чтобы policy не видела его labels и checkpoint criterion. Нужны как минимум:

- RM train, selection и time-disjoint audit intervals;
- сравнение с item-mean baseline и простым history baseline;
- MAE/Brier/calibration для bounded reward;
- error slices по item popularity, duration, history length, repeat/coldness;
- disagreement или второй RM seed/архитектура;
- доля policy actions вне RM support.

Низкий MAE на observed listens не доказывает корректность RM для unexposed catalog.

### 3.2 Observed next listen не равен relevance всех кандидатов

Остальные catalog items могут быть:

- не exposed;
- недоступны в момент времени;
- отфильтрованы продуктом;
- показаны, но проигнорированы;
- заменены organic action;
- повторно потреблены вне выбранного event stream.

Поэтому `eligible catalog` должен быть функцией только информации, доступной в timestamp. Обязательны проверки:

- item существовал/был eligible в момент test context;
- item не исключается скрытым future-derived фильтром;
- known/unknown target counts;
- repeat policy одинаков для всех policies;
- duplicate events и same-timestamp groups обработаны одинаково;
- candidate list hash/log сохранён для каждой evaluation run.

### 3.3 Multi-step state shift

Пять generated items подряд не означают пять реальных interactions. При простом append policy получает состояния, которые могут не существовать у пользователя. Нужно различать и отдельно маркировать:

1. teacher-forced one-step evaluation;
2. open-loop generated rollout;
3. closed-loop rollout в learned simulator;
4. фактические logged continuations.

Нельзя объединять их в один `NDCG@5` или один “long-term user” показатель. Обязательны diagnostics:

- RM/policy disagreement по шагам;
- novelty и repeat rate по шагам;
- state feature drift relative to logged states;
- reward uncertainty/OOD по шагам;
- separate H=1 and H=5 results.

### 3.4 Full catalog и support

Для полного каталога нужна точная спецификация:

\[
\mathcal C_t=\{i: i \text{ available and eligible at }t\}.
\]

Если фактически используется top-20k pilot catalog, это допустимо только как явно объявленный restricted-population/catalog feasibility experiment, как уже указано в плане. Нельзя переносить его результаты на Yambda full catalog.

Full softmax по миллионам item IDs может быть несовместим с RTX 5060 Ti 16 GB/48h. GPU/CUDA и throughput должны быть измерены до основного запуска. Ограничение вычислений не должно незаметно менять estimand с full catalog на sampled negatives.

### 3.5 KL regularization

Для KL-constrained/reference policy objective нужно зафиксировать направление и место применения:

\[
J(\pi)=
\mathbb E_{\tau\sim\pi}
\left[R(\tau)-\beta\sum_t
D_{KL}(\pi(\cdot|s_t)\|\pi_{SFT}(\cdot|s_t))\right].
\]

Нужно явно определить:

- forward или reverse KL;
- per-step или sequence-level normalization;
- exact full-catalog KL или sampled approximation;
- mask eligible/seen/special items до вычисления распределения;
- temperature и sampling rules;
- как выбирается `beta` и сохраняется ли его grid на selection.

Простой добавочный KL loss не обязательно эквивалентен указанному regularized RL objective. KL стабилизирует drift, но не устраняет confounding, reward-model error или simulator bias.

## 4. Разбиение, selection и независимость оценки

Рекомендуемый порядок:

1. global temporal train window;
2. RM train interval;
3. RM selection/audit intervals;
4. RL selection interval для early stopping и hyperparameters;
5. frozen final test interval;
6. по возможности later-period audit.

Границы должны быть атомарны для same-timestamp groups. Нельзя использовать future events при построении history, item popularity, vocabulary, eligibility или normalization scales.

Плановый 80/10/10 global temporal split пригоден как pilot при фиксированной sampled-user population, но необходимо сохранять:

- exact timestamps/quantiles;
- user/item counts per split;
- target coverage;
- same-user rolling-history convention;
- whether validation/test interactions are allowed in teacher-forced context (без weight update).

Последняя convention допустима, если она заранее объявлена и одинаково применяется ко всем моделям; это не должно быть названо cold-start evaluation.

Checkpoint нельзя выбирать по test reward, NDCG, RM или любому test-derived diagnostic. При отсутствии configuration, удовлетворяющей relevance floor, результат — failure/inconclusive, а floor не расслабляется после просмотра test.

## 5. Baselines и fair comparison

Минимальный comparison set:

1. popularity;
2. SFT GPT-2, deterministic top-K;
3. SFT GPT-2, одинаковый stochastic decoder;
4. SFT + reranking comparator;
5. KL-only/no-RL control;
6. simulator-on-policy RL;
7. H=1 и H=5 для policy без RL либо autoregressive control.

Все должны иметь одинаковые:

- contexts;
- eligible candidates;
- repeat/seen policy;
- tie breaking;
- temperature/sampling budget;
- inference compute budget.

Reranker не должен получать более узкий или более широкий candidate pool, чем RL, без отдельного обозначения. Для H=5 one-step top-10 и generated list — разные estimands и не должны сравниваться как одна метрика.

## 6. Формулы и источники

### 6.1 Petrov & Macdonald (2403.04875)

Корректные идеи:

- autoregressive Next-K можно представить sequential policy;
- list-level reward можно разложить на immediate/terminal rewards;
- PPO/value model нужны для actor-critic;
- teacher/reference checkpoint может использоваться для initialization/stability.

Ограничения и ошибки, релевантные этому проекту:

1. Разложение `R(h,g)=sum_i r(h,g^(i))` — определение, а не доказательство хорошего credit assignment.
2. Их NDCG discount использует `1/log(i+1)` вместо часто используемого `1/log_2(i+1)`; при фиксированном cutoff это постоянный масштаб, но definition должна быть объявлена.
3. В ILD pairwise normalization denominator `K(K-1)` при суммировании каждой unordered pair один раз даёт половину обычного среднего по unordered pairs; normalization нужно фиксировать.
4. В Eq. (8) MSE в переданном тексте записан с отрицательным знаком. При минимизации loss стандартная форма должна быть положительной:
   \[
   \sum_i(V_\Psi-\hat G_i)^2.
   \]
   Отрицательный знак как minimization loss максимизировал бы ошибку.
5. PPO в статье оптимизирует metric/reward, а не доказанный user feedback; это аналогично simulator-on-policy, но не causal user improvement.
6. Teacher/validation и published MovieLens/Steam результаты не переносятся автоматически на Yambda и миллионный catalog.
7. Autoregressive append generated items не моделирует acceptance feedback пользователя.

### 6.2 Matveev et al. (2607.02818v2)

On-policy REINFORCE reward-to-go формула

\[
\nabla J(\theta)=
\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_t\gamma^tG_t\nabla\log\pi_\theta(a_t|s_t)\right]
\]

является стандартной при корректной environment trajectory. Она не превращает logged data или generated fake states в user on-policy data.

Off-policy ratios

\[
\omega_{t:T}=\prod_{u=t}^{T}
\frac{\pi_\theta(a_u|s_u)}{\pi_b(a_u|s_u)}
\]

требуют known/consistently estimated behavior policy, absolute continuity и support. Clipping `min(c,omega)` уменьшает variance, но вводит bias; clipped estimator нельзя считать unbiased.

В переданном источнике Direct Method формально смешан с logged return: корректный DM должен использовать model/value estimate `mean_i \hat V^{\pi}(s_i)`, а не фактический `R(\xi_i)` из logged trajectories. Это важно не повторить в отчёте.

Sequential DR требует корректных `Q/V`, propensity/support и cross-fitting. Близость DR к behavior return не доказывает unbiasedness или superiority.

Источник сам признаёт, что test-time rollout selection и DM evaluation на одной feedback model дают только model-based improvement, не независимое evidence real policy quality. Этот caveat обязателен и для текущего проекта.

## 7. Минимально приемлемый RM protocol

RM должен предсказывать bounded completion reward с явной формулой и scale, вычисленной только на train. Отдельно анализировать:

- duration/played ratio;
- likes/dislikes/skip;
- repeat consumption;
- session continuation, если доступно;
- missingness и censoring.

Отчёт:

- item-mean/history-mean baselines;
- MAE, Brier/calibration, PR-AUC для sparse binary heads;
- per-user/per-item error distribution;
- popularity/duration/history-length slices;
- time-forward audit;
- RM uncertainty/disagreement;
- generated action support/OOD.

RM train/validation labels не должны включать final test. Если RM auditor не независим по данным или его параметры выбираются по RL test, он не является независимым audit.

## 8. Минимально приемлемый RL protocol

До запуска зафиксировать:

- horizon `H in {1,5}`;
- discount `gamma`;
- bounded reward и completion definition;
- baseline/advantage estimator;
- policy sampling temperature;
- KL direction, `beta`, normalization;
- eligible/seen/repeat mask;
- stopping and failure criteria.

Ablations:

- no KL, small/medium/large KL;
- reward-only, relevance-only, combined reward;
- H=1 vs H=5;
- terminal vs per-step reward;
- greedy vs stochastic rollout;
- repeat allowed vs no-repeat (если это продуктовый вопрос);
- independent RM seed/architecture.

Логировать:

- finite losses/gradient norm;
- entropy;
- actual KL to frozen SFT;
- reward variance;
- per-step reward;
- action support and RM OOD rate;
- coverage, duplicate rate, popularity concentration;
- seed-specific results.

## 9. Hypotheses и статистика из PLAN.md

H1 (RM): history-aware RM уменьшает test user-macro MAE против train-only item-mean на >=2% relative; paired user bootstrap 95% CI excludes no improvement. Это утверждение только для observed-listen population.

H2 (policy): KL-regularized simulator-on-policy RL увеличивает frozen RM sampled proxy return >=5% относительно SFT при non-inferiority NDCG@10 lower bound > -2%. Оба условия обязательны. Proxy gain без relevance floor не является recommendation-quality win.

H3 (exploratory): H=5 даёт больший model disagreement/context shift, чем H=1. Это diagnostic, не causal/confirmatory claim.

User bootstrap не заменяет training-seed uncertainty. При трёх seeds сообщать seed variance отдельно; не объединять их с bootstrap как независимые пользователи. При малой coverage или degenerate bootstrap — inconclusive.

## 10. Compute и stopping

До основного обучения проверить CUDA, GPU memory, forward/backward throughput и full-catalog timing. 48 часов — budget, а не обещание runtime.

Рациональная последовательность:

1. data audit и synthetic tests;
2. SFT + one-step RM;
3. throughput/memory benchmark;
4. baselines;
5. H=1 RL;
6. fixed evaluation;
7. только затем H=5 и дополнительные seeds.

Остановить эксперимент при leakage, insufficient target coverage, RM failure против простых baselines, nonfinite loss, memory overflow или отсутствии реализуемого full-catalog protocol. Не искать favorable result изменением test/split/relevance definition.

## 11. Что будет считаться честным итогом

Отчёт должен разделять четыре результата:

1. **Model-based:** вырос ли frozen RM completion score.
2. **Predictive offline:** изменился ли held-out next-listen NDCG/MAE.
3. **Simulator diagnostics:** насколько H=5 уходит в OOD/state shift.
4. **Causal/product:** отсутствует без randomized online test; не заявлять.

Фраза “RL улучшил рекомендации” допустима только в узком условном смысле: `на данном temporal test estimand RL сохранил/улучшил predictive relevance и повысил frozen model-based reward при объявленных ограничениях`. Фраза “повысил удовлетворённость пользователей” недопустима.

## 12. Подтверждённые ссылки

- Petrov, A. V., Macdonald, C. *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning.* arXiv:2403.04875. https://arxiv.org/abs/2403.04875
- Matveev et al. *Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction.* arXiv:2607.02818. Статус/версию проверить перед использованием как bibliographic evidence: https://arxiv.org/abs/2607.02818
- Williams, R. J. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning, 1992. https://doi.org/10.1007/BF00992696
- Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. https://arxiv.org/abs/1707.06347
- Schulman et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. https://arxiv.org/abs/1506.02438
- Jiang, N., Li, L. *Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.* ICML 2016. https://proceedings.mlr.press/v48/jiang16.html
- Dudík, M., Langford, J., Li, L. *Doubly Robust Policy Evaluation and Learning.* ICML 2011. https://proceedings.mlr.press/v15/dudik11a.html
- Ouyang et al. *Training language models to follow instructions with human feedback.* NeurIPS 2022. https://arxiv.org/abs/2203.02155
- Yambda dataset card: https://huggingface.co/datasets/yandex/yambda

## Final recommendation

Сохранять текущий scope как pilot model-based experiment. Не переименовывать его в user on-policy RL, не добавлять IPS/DR claims без logged propensities, не считать RM score независимым evidence. Сначала доказать воспроизводимость temporal one-step SFT/RM/baselines и корректность full eligible candidate protocol; только после этого масштабировать до H=5 и дополнительных seeds.
