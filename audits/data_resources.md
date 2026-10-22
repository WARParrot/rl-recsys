# Read-only аудит ресурсов и данных: Yambda next-item + reward model + RL

Дата аудита: 2026-10-22 (metadata/API проверены без скачивания parquet). Область: новый эксперимент GPT-2-like causal next-item, reward model и offline RL; это **не** feature search. Аудит не изменяет код/конфигурацию эксперимента. Единственный созданный артефакт — этот файл.

## 1. Итоговое решение

Yambda-50M является приемлемым минимальным датасетом для исследовательского offline proof-of-concept при строгом temporal protocol. Он не обеспечивает причинного доказательства, что RL улучшает пользовательскую удовлетворённость или retention: reward наблюдаем только после фактической экспозиции/прослушивания, а `is_organic` не является propensity score.

Полный каталог из 934,057 item и full-vocabulary GPT-2 head на 16 GB GPU технически возможен только с очень малым transformer и аккуратной memory strategy; обычный materialized logits `[batch, sequence, V]` — нет. В текущей среде GPU отсутствует (подтверждено пользователем), поэтому CPU environment подходит только для smoke tests/малых срезов, не для заявленного обучения.

### Критические стоп-проверки до обучения

1. Проверить реальную schema/типы/nulls и Parquet metadata на целевой машине; HF `first-rows` для multi-event в аудите завершился ошибкой `unsupported data type ... Dictionary(UInt32, Utf8View)`, поэтому sample rows здесь не подтверждены.
2. Зафиксировать семантику timestamp ties. Timestamp — 5-секундные bins; порядок событий внутри одного bin не идентифицируем.
3. Зафиксировать train/validation/test cutoff, gap, candidate universe и policy для repeated items/cold items до запуска.
4. Проверить, что item mapping, popularity, нормализация reward, фильтры и все агрегаты построены только по train.
5. Не считать рост reward model score причинным эффектом RL; обязательны baseline, KL/behavior constraint, coverage/repeat/concentration и temporal test.

## 2. Первичные официальные источники

- Dataset card: [Yandex Yambda-5B на Hugging Face](https://huggingface.co/datasets/yandex/yambda). Card указывает Apache-2.0, 4.79B interactions, 1M users, 9.39M tracks, multi-event/listen/like/dislike и audio embeddings. Версия/commit, видимый при аудите: dataset API SHA `dd6f3a19eef5866e346c3270e098baa641a44948`.
- Dataset API metadata: [HF dataset API](https://huggingface.co/api/datasets/yandex/yambda).
- Parquet conversion metadata (размеры, строки, memory estimate): [datasets-server parquet endpoint](https://datasets-server.huggingface.co/parquet?dataset=yandex/yambda).
- Aggregate size metadata: [datasets-server size endpoint](https://datasets-server.huggingface.co/size?dataset=yandex/yambda).
- Основная статья: [Ploshkin et al., “Yambda-5B -- A Large-Scale Multi-modal Dataset for Ranking And Retrieval”, arXiv:2505.22238v2](https://arxiv.org/abs/2505.22238v2), [API record](https://export.arxiv.org/api/query?id_list=2505.22238). Статья описывает Global Temporal Split и benchmark, но не устанавливает причинную валидность offline RL reward.
- Официальный benchmark guide: [benchmarks/README.md](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/README.md).
- Официальный split implementation: [timesplit.py](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/yambda/processing/timesplit.py).
- Официальные constants: [constants.py](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/yambda/constants.py).
- Официальные metrics: [metrics.py](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/yambda/evaluation/metrics.py).
- Официальный SASRec preprocessing: [data.py](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/models/sasrec/data.py).
- Официальная flat-to-sequential трансформация: [transform2sequential.py](https://huggingface.co/datasets/yandex/yambda/raw/main/benchmarks/scripts/transform2sequential.py).

## 3. Размеры и схема

HF size endpoint сообщил:

| Config | Parquet | rows | columns | estimated in-memory |
|---|---:|---:|---:|---:|
| flat multi-event 50M | 384,465,869 B (~367 MiB) | 47,790,449 | 7 | 423,552,771 B |
| sequential multi-event 50M | 437,501,893 B (~417 MiB) | 10,000 users | 7 | 952,165,854 B |
| flat multi-event 500M | 3,843,474,075 B | 480,255,564 | 7 | 4,247,843,373 B |
| flat multi-event 5B | 38,230,763,570 B | 4,785,859,887 | 7 | 42,308,888,040 B |

HEAD/redirect metadata для отдельных файлов:

- `flat/50m/listens.parquet`: content-length 369,110,332 B; URL root: https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/listens.parquet
- `flat/50m/multi_event.parquet`: content-length 384,465,869 B; URL root: https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/multi_event.parquet
- `sequential/50m/listens.parquet`: content-length 426,248,769 B; URL root: https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/listens.parquet
- `sequential/50m/multi_event.parquet`: content-length 437,501,893 B; URL root: https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/multi_event.parquet

Card statistics for Yambda-50M: 10,000 users, 934,057 items, 46,467,212 listens, 881,456 likes, 107,776 dislikes. “50M” is approximate and multi-event includes multiple event types.

Declared fields:

- homogeneous listen: `uid`, `item_id`, `timestamp`, `is_organic`, `played_ratio_pct`, `track_length_seconds`;
- other event files: `uid`, `item_id`, `timestamp`, `is_organic`;
- multi-event: previous fields plus `event_type`, optional playback fields;
- sequential: per-user lists of corresponding fields.

Card declares `uid`, `item_id`, timestamp as uint32 and `is_organic` as uint8; timestamp is binned into 5-second units and files are sorted by `(uid, timestamp)` ascending. `played_ratio_pct` and `track_length_seconds` are non-null only for listen events. These are card declarations; local sample-row validation remains required.

## 4. Timestamp ties and ordering risk

A timestamp bin is 5 seconds, so equal timestamps represent an interval, not a total order. Sorting by `(uid, timestamp)` does not identify which track occurred first inside that interval. The sequential conversion groups rows with `maintain_order=True`, preserving source-row order, but source-row order is not documented as event-time order for ties.

Required policy (choose and preregister one):

- exclude next-item transitions where adjacent timestamps are equal;
- or collapse/group same-bin events into a set/list and use set-aware evaluation;
- or use a deterministic tie-break only as an engineering convention and report that it is not causal/event-time order.

Do not claim that a within-bin transition is a true next action. Also test whether splitting flat data by timestamp can place same-bin events on different sides of a boundary. A gap measured in timestamp units must be interpreted as 5-second bins, not raw seconds.

## 5. Playback timing and reward labels

The card states:

- “listened” / `Listen_+` means more than 50% of track duration played;
- `played_ratio_pct > 100` is expected when rewind/replay causes total listened amount to exceed original length.

Therefore `played_ratio_pct * track_length_seconds` is a proxy for accumulated playback, not necessarily wall-clock time, and percent may exceed 100. It must not be silently clipped before analysis. Check distributions of >100, missing/non-listen fields, zero/very short durations and repeated user-item plays.

Possible reward labels should be declared separately:

- binary engaged listen: `played_ratio_pct > 50` (benchmark convention, not universal truth);
- bounded ratio: e.g. clipped or transformed ratio, with explicit treatment of >100;
- seconds proxy: ratio × duration, with duration outlier handling;
- like/dislike/multi-event reward.

A duration-based reward can game toward long tracks; ratio-based reward can game toward short tracks or replay artifacts. Report reward by duration decile and both ratio and absolute-seconds views. Do not combine listen and like events without specifying precedence when same user/item/timestamp has multiple events.

## 6. Official temporal benchmark behavior

The official `constants.py` declares:

- `GAP_SIZE = 30 minutes`;
- `VAL_SIZE = 1 day`;
- `TEST_SIZE = 1 day`;
- `LAST_TIMESTAMP = 26,000,000`;
- `TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE`;
- listen threshold = 50;
- ranked items = 100.

The official `timesplit.py` defines (timestamp units as stored):

- train before `test_timestamp - gap - val - gap` when validation exists;
- validation in `[test_timestamp - val - gap, test_timestamp - gap)`;
- test in `[test_timestamp, +inf)`;
- test/validation users restricted to train users;
- optional dropping of non-train items.

Important: this implementation is a useful reference, not automatically a valid protocol for the RL project. The code has no explicit tie handling, and `drop_non_train_items=False` by default. The project must decide whether unseen test items are evaluated (full catalog) or excluded (warm catalog), and report both where feasible.

The benchmark’s SASRec preprocessing filters listens at `played_ratio_pct >= 50`, constructs item mapping from the full input before split, and reserves index 0 for padding. Building the mapping from all timestamps would be leakage if copied literally for a strict experiment; construct train-only mapping or explicitly separate model vocabulary from evaluation catalog.

## 7. Candidate universe, filtering and leakage

The card FAQ states that not all test items appear in training, while test users do appear in training. Consequences:

- filtering unseen test items changes the estimand and can materially inflate warm-item metrics;
- full-catalog evaluation requires an explicit policy for item IDs unknown to the train model;
- candidate sets must be identical for baseline and RL;
- popularity, item frequency, track duration statistics, item embeddings, vocabulary and negative sampling distributions must be computed using train only unless the artifact is explicitly non-predictive and available at cutoff.

Required reports:

1. number/fraction of test targets absent from train;
2. number/fraction of test users absent from train (expected zero per FAQ, but validate);
3. metrics on warm items and full eligible catalog separately;
4. repeated-item policy (allow recommendation of known tracks or filter them);
5. candidate-generation recall if a two-stage candidate generator is used.

`is_organic` marks organic versus recommendation-driven interactions according to the card. This is valuable for slices, but it does not record complete exposure propensities or all non-clicked impressions. Observed listens are MNAR and selection-biased. Offline rank metrics are conditional evidence, not online impact.

## 8. Metric risks in official benchmark

The official metrics file itself records historical bugs:

- Recall denominator was formerly `num_positives` rather than `max(num_positives, k)`;
- NDCG was formerly computed as ratio of sums rather than mean per-user normalized DCG.

The Recall source comment mentions `max`, but the implementation uses `torch.clamp(num_positives, max=k)`, i.e. division by **min(n_positives, k)**, which is not standard Recall@K = hits / n_positives for a multi-target set. Avoid copying this silently; for single next-item targets the distinction disappears. MRR also needs a no-hit test: an argmax on an all-zero hit mask returns index 0 unless explicitly guarded.

The current source should be independently unit-tested. In particular, inspect the ideal-DCG path before relying on it for final claims; the downloaded source constructs `ideal_target_mask` but then calls `calc_dcg(target_mask)` rather than visibly using `ideal_target_mask`, which warrants a local regression test.

Use a single declared relevance definition and report macro per-user metrics, paired bootstrap confidence intervals, sample counts, item/catalog coverage, repeat rate, popularity concentration and diversity. Do not compare numbers generated by incompatible candidate universes or splits.

## 9. Full 934,057-item GPT-2 softmax memory estimate

Let `V=934,057`. Approximate output weight only (`V × d`):

| hidden d | fp32 weights | fp16/bf16 weights | Adam-style rough params+moments+grad budget |
|---:|---:|---:|---:|
| 128 | 456 MiB | 228 MiB | 1.34 GiB |
| 256 | 912 MiB | 456 MiB | 2.67 GiB |
| 384 | 1.34 GiB | 684 MiB | 4.01 GiB |
| 512 | 1.78 GiB | 912 MiB | 5.34 GiB |
| 768 | 2.67 GiB | 1.34 GiB | 8.02 GiB |

The optimizer column uses approximately 12 bytes per parameter (e.g. reduced-precision weights/gradients and FP32 moments), not a universal Adam footprint. FP32 parameters + FP32 gradients + two FP32 moments use 16 bytes/parameter: for d=768 the matrix alone is about 10.69 GiB during training. AMP/master-weight implementations and temporary optimizer buffers can increase this further. The estimate excludes transformer parameters, activations, allocator overhead, CUDA kernels and dataloader buffers. With tied input/output embeddings the parameter is shared, but full-vocabulary logits remain expensive.

Materialized logits `[B,V]` in fp16/bf16 are approximately:

- B=128: 228 MiB;
- B=512: 912 MiB;
- B=1024: 1.78 GiB;
- B=2048: 3.56 GiB;
- B=4096: 7.13 GiB.

For next-item training over sequence positions, `[B,T,V]` multiplies these by T. For B=128,T=128, fp16 logits alone are about 29 GiB. This rules out ordinary full-softmax training at such dimensions on a 16GB GPU.

Minimum plausible 16GB configuration: d=128–256, 2–4 layers, 4 heads, context 64–128, fp16/bf16, microbatch 8–32 with accumulation, activation checkpointing, SDPA/FlashAttention where supported, and either (a) carefully chunked full-vocabulary loss, (b) train-only reduced vocabulary with coverage reported, or (c) sampled/adaptive softmax with a full-catalog evaluation path. Sampled softmax changes the training objective and must not be presented as equivalent likelihood. Always benchmark one optimizer step before committing to a 48h run.

## 10. Environment and compute

Initial inspection of the host found:

- x86_64 Linux;
- 4 logical CPUs, Intel Pentium N3700 @ 1.60 GHz;
- RAM 7.2 GiB, about 3.2 GiB available at inspection;
- swap 4 GiB, already substantially used;
- no `nvidia-smi`, no `/dev/nvidia*`, no CUDA toolkit; PyTorch was absent from the system interpreter.

The parent thread subsequently confirmed GPU is unavailable **on this host** and installed a CPU PyTorch environment at `rl_recsys/.venv`. The user's separate target machine has a GPU; its exact model, VRAM, CUDA driver and throughput have not been audited. This environment is suitable for import/unit/split/metric tests and very small smoke runs, not 50M training or RL rollout within a reliable 48h budget. Do not install packages or download the dataset on this host as part of this audit.

### Portability handoff (requirements, not implementation audit)

The standalone Git repository is `/root/t-lab/rl_recsys`; the main GPU dependency lock must not pin a CPU-only PyTorch wheel/index based on this host's `.venv`. Keep CPU smoke-test setup explicitly separate. GPU installation instructions should select a supported PyTorch CUDA wheel for the user's driver/platform rather than copying the old benchmark's CUDA 11.8/implicit restrictions (those concern its optional baseline, not GPT-2 training).

A portable preflight must report Python/PyTorch versions, `torch.version.cuda`, CUDA availability, GPU model and VRAM, driver visibility, supported precision, disk/RAM headroom, selected package index and actual installed wheel. It should fail clearly for a requested GPU run with CPU-only torch or no visible GPU, and run a small device allocation + forward/backward/optimizer step using the intended precision, vocabulary and microbatch. Target-machine throughput and peak allocated/reserved VRAM must determine the final 48h schedule; installation success alone is not evidence of feasibility. Code/lock implementation review is deferred until requested.

## 11. Minimal defensible experiment

1. Start with `flat/50m/listens.parquet` for a clean playback target; use multi-event only if likes/dislikes/unlikes are part of the declared reward.
2. Inspect schema and row-group statistics using HTTP range/Parquet metadata on the target machine; avoid loading full data into pandas. Validate counts against HF API.
3. Define a global temporal cutoff, validation window and gap. Apply per-user causal history cutoff. Exclude or group equal-timestamp transitions.
4. Build train-only item vocabulary and separately record eligible full-catalog IDs; quantify cold-item impact.
5. Train a small next-item baseline first. Retain popularity and simple sequential baseline as sanity comparators.
6. Train reward model only from train histories/events. Evaluate calibration/ranking on a later temporal holdout, including duration and organic/recommended slices.
7. RL should be conservative: KL penalty or behavior regularization to baseline, bounded reward, no repeated-item loophole unless intentional, and short-horizon rollout first. Compare one-step and multi-step generation; generated items are not observed user actions, so feeding them back creates off-policy/model-extrapolation risk.
8. Primary report: Recall@K and corrected per-user NDCG@K; secondary: reward, coverage, repeat rate, item concentration, duration distribution, organic slice and uncertainty. Same candidates and test set for baseline/RL.
9. Declare each hypothesis supported/not supported/inconclusive. An increase in predicted reward with lower relevance, coverage collapse or reward-model exploit is a failure mode, not success.

## 12. Unresolved/undocumented risks

- Exact global timestamp origin/unit semantics beyond 5-second bin declaration are not established locally.
- Equal-bin source ordering and duplicate event semantics are not established.
- Exposure logs are incomplete; `is_organic` is not a complete policy propensity.
- Test item cold-start policy is a choice, not an innocuous filter.
- Benchmark code contains documented historical metric bugs and requires independent tests.
- Multi-event same-time event precedence is unspecified.
- `played_ratio_pct > 100` makes naive ratio-to-seconds reward potentially exploitable.
- Full-catalog GPT-2 output and RL sampling can dominate memory/compute even when model parameter count appears small.
- Offline reward model is vulnerable to shortcut learning (duration, popularity, exposure channel, replay) and adversarial policy shift.
- Multi-step generated trajectories are synthetic; a reward model evaluated on them is extrapolating outside its logged support.

## 13. HTTP/range audit note

Metadata and HTTP HEAD/redirects were queried only. No event parquet was fully downloaded. HF returned signed Xet URLs with ETags/content lengths and advertised range-related headers. A safe follow-up on a GPU/analysis host is to fetch only the Parquet footer via HTTP Range, then inspect row-group statistics and schema with a Parquet reader; verify that the reader handles the HF dictionary/`Utf8View` encoding before trusting server-side first-row APIs.
