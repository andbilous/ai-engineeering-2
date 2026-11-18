# Заняття 12. Власний MCP-сервер

Магазин настільних ігор «Кубик» хоче асистента, який відповідає клієнтам про ігри, ціни й наявність і показує статус їхніх замовлень. Каталог, замовлення й умови магазину живуть в MCP-сервері `server.py`, а асистент (`host.py`) підключається до нього за протоколом і нічого не знає про магазин наперед. Дані вигадані.

## Встановлення

```bash
cd lesson-12-mcp
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # mcp 2.x: FastMCP тут називається MCPServer
cp .env.example .env                # вставте свій OPENROUTER_API_KEY
```

Без ключа працюють перевірки 1–7. Повний прогін 14 задач коштує близько $0.003.

## Що зробити

Над кожним TODO є коментар Goal / Why / Done when.

| TODO | Файл | Що |
| --- | --- | --- |
| 1 | `starter/server.py` | інструмент `search_products`: схема з типів, опис з docstring |
| 2 | `starter/server.py` | інструмент `get_order`: лише замовлення поточного клієнта |
| 3 | `starter/server.py` | ресурс `shop://policy` |
| 4 | `starter/host.py` | `mcp_spec`: інструмент MCP → формат chat API |
| 5 | `starter/host.py` | `call_mcp`: виклик на сервері, помилки повертаються моделі текстом |

## Як перевірити себе

```bash
python starter/check.py                                      # перевірки 1–8: PASS, FAIL або TODO
python starter/host.py --ask "Чи є у вас Каркасон?"          # кроки й відповідь
python starter/host.py --eval                                # 14 задач, результат у output/
python starter/host.py --eval --customer C-42                # той самий сервер для іншого клієнта
```

Після TODO 1–3 мають пройти перевірки 1–4 і 7, після TODO 4–5 решта. Останній прогін порівняйте з першим: які задачі змінили результат і чому.

`data/` змінювати не треба.
