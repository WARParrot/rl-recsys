# Completion-only follow-up: результат

**Статус:** exploratory reuse, отрицательный результат; это отдельный follow-up, не новая независимая проверка.

Артефакт: `completion-20261025-203706-879.zip`
SHA-256: `1de1edd6d02f4f8692268234f5f1269e3056ab011c3f3c4ee922cfb745949071`
Код: `181c4c235390f3aca69336ea45f2c695842042eb`
Данные и SFT: переиспользованы из исходного research-run; `prepared.npz` SHA-256 `b30da6213320d94b8d3acc6f18e77a33b6e5f68c5db6b2a4c7e15f759888397e`.

Все **109/109** записей внутреннего `SHA256SUMS.json` проверены. Запуск выполнен на Windows/CUDA; новые RM обучались около **945 секунд**, evaluation занял **62 секунды**. GPU inference и обучение RL не выполнялись, потому что reward gate остановил reward-зависимые arms.

## Вопрос и протокол

Проверялась гипотеза:

> Если убрать auxiliary seconds-loss и обучать reward model только completion, станет ли completion-модель достаточно точной для RL-оптимизации?

Сохранены те же prepared data, каталог, временные границы, SFT и sampler. Обучены новая history-aware RM и learned item-only comparator. Completion-only objective использовал MSE; checkpoint выбирался по user-macro completion MSE на early calibration. Поздний audit использовался только для gate/описательной проверки. Gate остался прежним: point completion MAE должен быть минимум на 2% лучше fit item-mean и лучше константы.

Периоды calibration и test уже изучались в предыдущих запусках. Все новые выводы имеют статус `exploratory_reuse`; они не являются подтверждающими и не причинными.

## Reward model

Поздний audit: **3 065 примеров, 605 пользователей**.

| Модель | Completion MAE | Item-mean MAE | Изменение к item-mean | 95% CI изменения |
|---|---:|---:|---:|---:|
| Историческая joint completion/seconds RM | 0,347538 | 0,348721 | −0,339% | [−3,379%; +2,720%] |
| Новая completion-only RM | **0,349361** | 0,348721 | **+0,183%** | [−2,794%; +3,064%] |

Completion-only модель не достигла требуемых 2% и точечно немного хуже item-mean. Она также хуже исторической joint RM на этом audit-периоде примерно на 0,52% MAE. Поэтому completion gate **не поддержан**, а RL, RL+CE и reranking не запускались.

Seconds output в completion-only режиме был исключён из loss и помечен как необученный; seconds reward не допускался.

На ранее просмотренном test новая RM имела MAE `0,343813` против item-mean `0,347756`, то есть −1,134%; paired user-bootstrap CI `[−3,503%; +1,297%]`. Этот результат не может отменить failed late-audit gate.

### Обучение и переобучение

Для completion-only RM лучший early-calibration checkpoint был на шаге 100: MSE `0,162594`. К шагу 3000 train loss снизился до `0,023876`, но calibration MSE вырос до `0,236266`. Это согласуется с переобучением после раннего checkpoint, но выбранный checkpoint был ранним; провал late audit не объясняется использованием финальных весов.

Выбор checkpoint по MSE и допуск по MAE измеряют разные свойства. MSE ближе к conditional mean, необходимому для expected reward, а MAE ближе к median-oriented ошибке. Поэтому текущий point-gate остаётся консервативным операционным правилом, но не гарантией качества counterfactual expected reward.

## Policy controls

Из-за failed gate обучены только continued-SFT controls от общей SFT/RM/data для optimization seeds 42, 43 и 44. RL, RL+CE и reranking отсутствуют в результатах и не должны интерпретироваться как нулевой эффект: они **не были допущены к обучению**.

| Seed | Eligible NDCG@10 | Изменение к SFT | 95% CI | Expected RM proxy change |
|---|---:|---:|---:|---:|
| 42 | 0,029109 | +1,58% | [−2,52%; +6,07%] | +0,0120% |
| 43 | 0,029284 | +2,20% | [−1,66%; +6,57%] | +0,0116% |
| 44 | 0,028910 | +0,89% | [−3,13%; +5,10%] | +0,0123% |

Все интервалы NDCG включают ноль. Изменение proxy мало и относится к той же fitted RM, а не к реальному user utility. Выбирать seed 43 по test нельзя.

## Вердикты

| Гипотеза | Вердикт | Основание |
|---|---|---|
| Completion-only loss даст RM, проходящую gate | **Не поддержана** | Late MAE +0,183% к item mean при требовании −2%; CI включает ноль |
| Удаление seconds-loss улучшит RM относительно joint RM | **Не поддержана** | Completion-only late MAE 0,349361 против 0,347538 у joint RM; сравнение описательное |
| Continued-SFT улучшит relevance устойчиво по seeds | **Неопределённо** | Все три point estimates положительны, но все CI включают ноль |
| RL+CE улучшит reward–relevance trade-off | **Не проверена** | RL-dependent arms были остановлены gate; данных для сравнения нет |

## Интерпретация

Результат не показывает, что RL+CE не работает. Он показывает, что при текущих logged listens, каталоге, истории и протоколе **ни joint RM, ни completion-only RM не дали достаточно надёжного completion-сигнала для заранее разрешённой policy optimization**.

Смена joint objective на completion-only не решила проблему. Возможные, но не установленные механизмы:

- слабая контекстная предсказуемость completion в позднем временном периоде;
- exposure/selection bias logged listens;
- mismatch между RM MAE/MSE и ranking качества кандидатов;
- калибровочный сдвиг и ошибочные высокие scores в policy-relevant хвосте;
- ограниченная статистическая мощность и reused audit.

Нельзя из этого запуска заключать, что seconds-loss был причиной провала, что reward hacking доказан или что completion-only policy была хуже: policy arms не запускались.

## Следующий эксперимент

Не ослаблять gate и не перебирать гиперпараметры до получения положительного test результата. Наиболее полезная следующая ветка должна менять **информацию для reward**, а не только loss:

1. использовать несколько независимо обученных RM на одном fixed catalog/data;
2. проверить, предсказывает ли disagreement абсолютную ошибку на отдельном observed-label audit;
3. только при поддержанном disagreement-gate запускать mean и `mean - kappa * std` reward;
4. сравнить с item-mean, item-only, reranking и continued-SFT;
5. для сильного вывода нужен новый незатронутый временной test.

Retention, acceptance/skip transitions и causal user impact остаются отдельными задачами. Ни один результат этого follow-up не доказывает удовлетворённость, удержание или онлайн-пользу.
