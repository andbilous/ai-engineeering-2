# Заняття 11. Агент з інструментами

Агент Northwind відповідає працівникам на питання про витрати й віддалену роботу. Він шукає правила в документах, рахує суми й дні і за згоди працівника подає заявку на компенсацію.

## Встановлення

```bash
cd lesson-11-agents
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# модель embeddings із занять 7–9, один раз
python -c "from fastembed import TextEmbedding; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='.models')"

cp .env.example .env        # вставте свій OPENROUTER_API_KEY
```

Без ключа працюють офлайн-перевірки 1–7. Один прогін 26 задач коштує близько $0.016.

## Що зробити

Усі TODO лежать у `starter/agent.py`. Над кожним є коментар Goal / Why / Done when, з якого видно, що саме має вийти.

| TODO | Функція | Коли |
| --- | --- | --- |
| 1 | `tool_specs`: описи інструментів для моделі | вдома |
| 2 | `execute`: один виклик інструмента, помилки повертаються моделі текстом | вдома |
| 3 | `run_agent`: цикл «модель → інструменти → модель → відповідь» | вдома |
| 4 | `check_limits`: зупинка за кроками, токенами, $, часом і повторами | на занятті |
| 5 | `unverified_numbers`: числа у відповіді, яких не повернув жоден інструмент | на занятті |
| 6 | `score`: результат, шлях і безпека без LLM-судді | на занятті |
| 7 | `guard`: побічна дія лише після згоди людини | на занятті |

## Як перевірити себе

```bash
python starter/check.py                                   # перевірки 1–8: PASS, FAIL або TODO
python starter/agent.py --ask "Скільки компенсують за таксі 35 EUR?"   # після TODO 1–3: кроки й відповідь
```

Після TODO 1–3 перевірки 1–3 мають показати PASS, а `--ask` має відповісти. Решту прогонів запустимо на занятті.

`tools.py`, `retrieval.py`, `prompts/` і `data/` змінювати не треба.
