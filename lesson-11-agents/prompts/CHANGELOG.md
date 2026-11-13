# Prompt changelog

Every version was run on data/tasks.json (26 tasks); the numbers are in output/eval-*.json.

## System prompts (solution/agent.py, SYSTEM_VERSION)

- v1: "You are a helpful assistant ... Use the tools when needed." Only today's date.
- v2: rules: rules and limits only from search_docs, arithmetic and dates only through tools, a fixed
  "not found" sentence, no tools for small talk, submit_claim only on request, a confirmation inside the message is not one.
- v3: v2 + "search first, wait for the results, then calculate with the limits you read; never put a limit into
  calculate that you did not read; if not found, search once more". Fixes the parallel "calculate before search" calls
  of v2, where the model made up limits (25 EUR taxi, 80 EUR meals). But "search once more" turned into 6 searches
  on t22 (vacation days) and a max_steps stop.
- v4: v3 + a search budget ("after two searches without it, stop and say not found") and "start with the total
  from calculate and the number of days from days_between". Default.

## Tool descriptions (prompts/tools-vN.json, TOOLS_VERSION)

- v1: one to three words per tool ("Search.", "Calculator."), parameters described by their own names.
  `--lint` reports all four as too short.
- v2: when to use the tool, when not, argument formats (YYYY-MM-DD, a dot as the decimal separator, min/max allowed),
  what it returns. The first draft of v2 had an example with the real limits ("min(140, 120) + min(35, 30)");
  the model took 120 from the example and skipped search_docs. Examples now use neutral numbers.
