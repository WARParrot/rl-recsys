# Completion-only follow-up: отдельный exploratory experiment

**Фактические результаты доступны в [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).** Ниже оставлены исходные инструкции и протокол; запуск завершился штатно, но completion-only RM не прошла gate, поэтому RL-dependent policy arms не запускались.

Это **новое обучение reward**, не повтор SFT и не новый независимый test. Основные численные таблицы README относятся к старому pilot и не являются результатами этого запуска.

## Запуск на Windows (cmd.exe)

Сохраните исходный полный каталог `research-20261025-150300-430` со всем содержимым. Нужен именно каталог, а не ZIP: внутри должны быть `sample_42\data\prepared.npz`, `audit.json`, `sample_42\run\sft.pt`, `SFT_READY.json`, provenance/training, а также исходный `sample_42\followup` с manifest, provenance и seed_42 RM/компаратором/аудитом. Не меняйте старые manifests. Повторно скачивать parquet не нужно.

Из корня репозитория, в существующем Python 3.11/CUDA окружении предыдущего запуска:

```bat
git pull --ff-only origin main
run_completion_windows.bat "C:\path\to\research-20261025-150300-430"
```

Замените только `C:\path\to\...` своим реальным путём к **исходному корню suite** (не `sample_42`). Выход по умолчанию — новый timestamp-каталог `artifacts\completion-*`. Опциональный второй аргумент — новый выходной путь. Пути с пробелами поддерживаются; существующий output/ZIP не перезаписывается. Launcher использует уже установленную `.venv\Scripts\python.exe`, ничего не устанавливает и не запускает старый default suite.

Эквивалентный Python CLI, в том числе для заранее выбранного меньшего **одинакового** бюджета всех policy arms:

```bat
.venv\Scripts\python.exe -u -m rl_recsys.completion_study --parent "C:\path\to\research-20261025-150300-430" --output "artifacts\completion-fixed" --steps 1000 --max-training-hours 4.5
```

Не менять бюджет после просмотра новых метрик. По умолчанию training subprocess принудительно останавливается через **4,5 часа**, evaluation имеет 90-минутный внутренний лимит (95 минут внешний). Timeout — ошибка/незавершённый эксперимент, а не успешное досрочное обучение; сохраняются доступные файлы и failure archive. Запас на упаковку/передачу/отчёт нужен отдельно. Архивирование не имеет временного лимита. Прежние ~21 минута относятся к suite **без RL**, поэтому не оценивают время admitted RL. Windows/CUDA runtime этого нового пути здесь не измерялся.

## Фиксированный контраст

Гипотеза: исключение auxiliary seconds-задачи может улучшить completion modeling. Механизм **не установлен**, положительный результат не требуется.

- Те же prepared data, каталог, временные границы, fit/early/late/selection/test, SFT, архитектуры и sampler. Одно новое history-RM и один новый item-only comparator; исходный joint baseline остаётся замороженным.
- RM и item-only: **completion MSE** при обучении, **user-macro completion MSE на early calibration** при выборе checkpoint. MAE не является objective: MSE нацелен на условное среднее, необходимое для expected-reward оптимизации.
- Для совместимости сохранена двухвыходная архитектура. Seconds-label/output исключены из loss, secondary output не обучен как предиктор; общий trunk изменяется, поэтому это не буквально неизменный численный выход. Seconds-метрики явно помечены необученными, seconds reward запрещён gate.
- RM бюджет наследуется из оригинального config (в возвращённом запуске 3000 обновлений), eval_every=50; исходный selected RM был step100, не final3000. Новый objective относится и к learned item-only.
- Прежний point-gate: completion MAE минимум на **2%** лучше fit itemmean и лучше fit constant. CIs, RMSE, calibration/tails/slices остаются в audit. Gate не гарантирует точность контрфактической ожидаемой полезности. При неудаче не ослабляем threshold, не ищем новые настройки и не используем благоприятный test для обхода gate.
- При failed gate выполняется только continued-SFT; при admitted gate — continued-SFT, completion H1 RL, RL+CE 0.1/1 и rerank grid 0/0.1/1/10. Beta=.01, T=1, 1000 обновлений на trained arm по умолчанию, seeds42/43/44 с **одним общим RM/SFT/data**. Это conditional optimization variability, не три независимых RM.
- Фактический запуск: failed gate для всех reward-dependent arms, так что были обучены только continued-SFT controls. Подробная таблица и выводы: [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).
- На reused test completion-only MAE=`0.343813` против item-mean `0.347756` (−1.134%; CI [−3.503%; +1.297%]); этот test score не используется для обхода gate.
- На top10% predicted completion (509 events/231 users) event-mean prediction `.88499`, observed `.85703` (~2.80pp overprediction). Top1%: 51 events/34 users, insufficient support. Это только logged-target audit, не label качества непоказанных действий.
- Все training/selection завершены до test. Selection floor и frozen reward одинаковы для arms; failed floor сохраняется как failure. Evaluation использует тот же новый completion-RM для всех новых политик. Абсолютные policy scores разных RM не следует сравнивать как независимую истину.

## Артефакты и передача

Новый output содержит:

- `inputs/data`, `inputs/run`: lossless копии исходных prepared arrays и SFT с оригинальными hash-bound metadata;
- `historical-joint`: исходные selected RM/item-only weights, curves, late audit и provenance/manifest, без переобучения; `IDENTITY.json` связывает копии hashes. Исторический followup manifest остаётся **исходным**, с ссылками и на не скопированные policy artifacts: это excerpt baseline, не полный старый suite;
- `followup`: новые objective/config/source hashes, **явный `source_mode=historical_reanalysis` только для импорта old-code SFT**, старые и новые source identities, все gate/selection/curve/checkpoint артефакты;
- `controls-evaluation`: явный exploratory test, paired comparisons, eligible/all-target populations, behavioral metrics и per-example массивы. Здесь source check остаётся **current** относительно нового training provenance, не отключается;
- `reward-comparison.json`: компактные исходный joint и новый completion-only late-audit со сравнением с itemmean/item-only, решение gate и ограничения. Это не новый paired CI joint-vs-completion; отдельные paired baseline CIs есть в каждом audit;
- `study.json`, `logs/train.log`, `logs/evaluate.log`: статус, elapsed и lossless subprocess logs; `source/` в ZIP фиксирует код.

Отправить `<output>.zip` и `<output>.zip.sha256.json`. ZIP64 содержит все новые файлы, selected+final policy weights, скопированные prepared/SFT/RM, исходники; каждый member reread и SHA256-проверяется. Нужен запас диска под копии и ZIP. Автоматической отправки/resume нет. При лимите загрузки сначала можно отправить `study.json`, `reward-comparison.json`, `controls-evaluation/results.json` и новые `reward_audit.json`/`reward_curves.json`; затем веса/arrays отдельными архивами с исходными путями. **Не обрезать или переписывать оригинальные manifests**: неполный upload не становится полным bundle. Для независимого воспроизведения нужны также большие файлы.

## Основание и границы выводов

Возвращённый run `5c0cc00`: все 105 исходных manifest-файлов восстановлены и сверены, хотя переданный ZIP был усечён. Один общий joint RM не прошёл late gate: completion MAE **0.347537686** против itemmean **0.348720769**, изменение **−0.3393%**, paired95% **[−3.3789%; +2.7199%]**. Поэтому новых RL/RL+CE/rerank там **не было**. Выбран лучший early checkpoint, а не переобученный final. Благоприятный test MAE не отменяет failed gate. Эти значения не подменяют старые pilot-таблицы README.

Новый запуск остаётся **exploratory_reuse**: late audit и test уже изучены. Отказ от seconds — новая гипотеза после диагностики, не confirmatory preregistration. Даже проход gate и рост proxy не означают реальную пользу пользователям. Нужны отдельные анализ reward–relevance, Recall/NDCG, coverage/HHI/repeats/duration, matched-control contrasts и ограничения observational labels. Ensemble, retention и H5 здесь не добавляются.
