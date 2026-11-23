# Підключення до вашого агента

Інструкції пишемо один раз в `AGENTS.md` у корені проєкту. Шаблон — `AGENTS.template.md` поруч; у корінь його кладіть лише на TODO 3, щоб на перших кроках агент працював без інструкцій. Більшість агентів читає його самі, решті потрібен один короткий файл з посиланням. Перевірено за документацією станом на листопад 2026; формати змінюються, тож якщо щось не спрацювало, звіряйтеся з документацією свого інструмента.

## Інструкції проєкту (TODO 3)

| Інструмент | Що читає | Що зробити |
| --- | --- | --- |
| Codex CLI, Cursor, GitHub Copilot, Cline, Windsurf, Zed, Aider | `AGENTS.md` | нічого |
| Claude Code | `CLAUDE.md`; `AGENTS.md` лише якщо `CLAUDE.md` немає | нічого, або створіть `CLAUDE.md` з одним рядком `@AGENTS.md` (імпорт), якщо хочете додати щось лише для Claude |
| Gemini CLI / Antigravity CLI | `GEMINI.md` | у `.gemini/settings.json`: `{"context": {"fileName": ["AGENTS.md"]}}` |

## MCP-сервер (TODO 5)

Приклад підключає браузерний сервер Playwright: агент відкриває сторінку, клацає і читає її вміст. Потрібен Node.js 18+. Файли в `mcp/` скопіюйте туди, куди вказано, і перезапустіть агента.

| Інструмент | Куди покласти | Файл-приклад |
| --- | --- | --- |
| Claude Code | `.mcp.json` у корені проєкту | `mcp/mcp.json` |
| Cursor | `.cursor/mcp.json` | `mcp/mcp.json` |
| VS Code + Copilot | `.vscode/mcp.json` | `mcp/vscode-mcp.json` (ключ `servers`, а не `mcpServers`) |
| Cline | панель MCP Servers → Configure, файл `cline_mcp_settings.json` | вміст `mcp/mcp.json` |
| Codex CLI | `~/.codex/config.toml` | `mcp/codex-config.toml` |

Перед підключенням будь-якого іншого сервера подивіться, хто його опублікував і що він може робити. MCP-сервер виконується з вашими правами.

## Skill (TODO 6)

Шаблон у `skills/business-rule/SKILL.md`. Формат відкритий (agentskills.io), тека та сама, лише корінь різний:

| Інструмент | Куди покласти теку `business-rule/` |
| --- | --- |
| Claude Code | `.claude/skills/` |
| Cursor, Copilot у VS Code, Codex CLI та інші | див. розділ про skills у документації інструмента; більшість шукає `.agents/skills/` або власну теку на кшталт `.cursor/skills/`, `.github/skills/` |

Якщо ваш інструмент skills не підтримує, покладіть процедуру в окремий файл `docs/business-rule.md` і додайте в `AGENTS.md` рядок «перед додаванням бізнес-правила прочитай `docs/business-rule.md`». Ідея та сама: процедура вантажиться лише тоді, коли потрібна.
