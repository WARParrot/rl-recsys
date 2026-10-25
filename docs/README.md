# Документация проекта

Начните с [README в корне](../README.md): там кратко описаны задача, результаты и запуск. [Полный исследовательский отчёт](reports/STUDY.md) содержит настройки, все таблицы, диагностику, ограничения и источники.

## Что читать

| Если нужно… | Документ |
|---|---|
| Быстро посмотреть результаты | [README](../README.md#результаты) |
| Изучить полный отчёт | [reports/STUDY.md](reports/STUDY.md) |
| Прочитать краткую сводку на английском | [reports/RESULTS.md](reports/RESULTS.md) |
| Посмотреть результат отдельного completion-only запуска | [reports/COMPLETION_RESULTS.md](reports/COMPLETION_RESULTS.md) |
| Повторить вычисления на исторических весах | [guides/REANALYSIS.md](guides/REANALYSIS.md) |
| Запустить completion-only follow-up | [guides/COMPLETION_FOLLOWUP.md](guides/COMPLETION_FOLLOWUP.md) |
| Проверить предсказания ансамбля | [guides/ENSEMBLE_AUDIT.md](guides/ENSEMBLE_AUDIT.md) |
| Изучить протокол следующего эксперимента | [research/EXPERIMENT_PROTOCOL.md](research/EXPERIMENT_PROTOCOL.md) |
| Разобраться в реализации Experiment A | [research/REWARD_DESIGN.md](research/REWARD_DESIGN.md) |

**Статус экспериментов важен:** исторический joint-RM pilot и completion-only follow-up выполнены. Для Experiment A реализован код, но завершённый полный запуск не заявляется. Инструкции и планы сами по себе не являются результатами эксперимента.

## Разделы

- **`reports/`** — дополнительные отчёты о выполненных запусках.
- **`guides/`** — инструкции запуска и проверки. Все команды выполняются **из корня репозитория**, а не из каталога документа.
- **`research/`** — протокол и описание исследовательского набора Experiment A, а также границы будущего Experiment B.
- **`archive/`** — история проекта: [старый отчёт](archive/REPORT.md), [исходный план](archive/PLAN.md), [план одного запуска](archive/ONE_RUN_PLAN.md), [изменения после аудита](archive/RESEARCH_UPGRADE.md) и [список улучшений](archive/IMPROVEMENTS.md). Эти документы сохранены для контекста; актуальные выводы и команды находятся в основном README и инструкциях выше.

Отдельно от документации:

- [evidence/reanalysis-e3cf359](../evidence/reanalysis-e3cf359/README.md) — опубликованные выдержки из результатов, графики и manifests. Большие веса и массивы не хранятся в Git.
- [audits/](../audits/) — исторические независимые проверки методологии и ресурсов. Их замечания относятся к состоянию проекта на момент аудита.

## Где искать код

| Задача | Модули |
|---|---|
| CLI, конфигурация и проверка окружения | `rl_recsys/cli.py`, `config.py`, `preflight.py` |
| Подготовка данных | `rl_recsys/data.py` |
| Модели и базовое обучение | `rl_recsys/models.py`, `experiment.py` |
| Метрики и диагностика | `rl_recsys/evaluate.py`, `diagnostics.py`, `reward_audit.py` |
| Повторная оценка исторических весов | `rl_recsys/reanalysis.py` |
| Дополнительные эксперименты | `rl_recsys/followup.py`, `followup_evaluate.py`, `completion_study.py`, `suite.py` |
| Аудит ансамбля и сохранение артефактов | `rl_recsys/ensemble_audit.py`, `artifacts.py` |

Python-пакет остаётся в `rl_recsys/`, тесты — в `tests/`, конфигурации — в `configs/`. Windows-скрипты оставлены в корне, чтобы существующие команды запуска продолжали работать.

Ранее дополнительные Markdown-документы лежали в корне. Теперь они сгруппированы здесь; имена файлов сохранены, относительные ссылки обновлены. Исторические release-архивы и содержимое `evidence/` не переписывались.
