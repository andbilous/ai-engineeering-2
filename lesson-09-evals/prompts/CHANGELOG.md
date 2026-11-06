# Prompt changelog

## Bot prompts (rag.py, PROMPT_VERSION)

- v1: lesson 8 starter.
- v2: lesson 8 after the lesson: "discontinued" is an answer, the newer source wins, numbers copied exactly. Baseline.
- v3: "be more helpful": may complete the answer with common practice. Pass stays the same, groundedness drops:
  the gate must stop it.

## Judge prompts (solution/evals.py, JUDGE)

Calibrated on 90 answers labelled by hand (data/labels.json): `python solution/evals.py --calibrate --judge-prompt judge-vN`.
Agreement with people, gpt-4o-mini as the judge:

| prompts | correct | grounded | what changed |
|---|---|---|---|
| judge-v1 | 0.94 | 0.92 | first version, before calibration |
| judge-v2 | 0.88 | 0.96 | grounded: claims only from <answer>, an added detail is a claim of its own; correct: "a condition that changes the meaning fails", which made it fail right answers with extra true details |
| judge-v3 | 0.92 | 0.96 | correct: judge only the key fact the question asks for, extra details are fine unless they contradict |
| judge-v3, q13 reference fixed | 0.99 | 0.94 | the reference of q13 said "up to 25 kg" and left out 15 kg for women: 6 of 7 disagreements were the testset's fault |

The grounded numbers of v2 and v3 use the same prompt: 0.96 and 0.94 is the judge's own noise between two runs.
