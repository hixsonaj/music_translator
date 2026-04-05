# Lyric Quality Agent System

Multi-agent pipeline using Fetch.ai uAgents that negotiates translated lyrics
to match the original's syllable count, rhyme scheme, and timing.

## File structure

```
lyric_agents/
  models.py           — Shared data models (LyricLine, CritiqueResult, etc.)
  translation_agent.py — LLM-backed translation + revision
  syllable_critic.py  — Checks syllable count (±1 tolerance)
  rhyme_critic.py     — Checks rhyme scheme preservation
  timing_critic.py    — Checks line fits within Whisper segment duration
  orchestrator.py     — Coordinates the negotiation loop
  backend.py          — FastAPI entry point (feeds lines in, collects results)
  requirements.txt
```

## Startup order

Run each agent in a separate terminal. Grab the printed agent address after startup.

```bash
# Terminal 1 — Translation agent
python translation_agent.py
# → prints: agent address = agent1q...  ← copy this

# Terminal 2 — Syllable critic
python syllable_critic.py

# Terminal 3 — Rhyme critic
python rhyme_critic.py

# Terminal 4 — Timing critic
python timing_critic.py

# Terminal 5 — Orchestrator (set addresses from above)
TRANSLATION_ADDR=agent1q... \
SYLLABLE_ADDR=agent1q... \
RHYME_ADDR=agent1q... \
TIMING_ADDR=agent1q... \
python orchestrator.py

# Terminal 6 — FastAPI backend
ORCHESTRATOR_ADDR=agent1q... \
OPENAI_API_KEY=sk-... \
python backend.py
```

## Quality targets per critic

| Critic    | Target                                         | Tolerance |
|-----------|------------------------------------------------|-----------|
| Syllable  | Translated syllable count = original count     | ±1        |
| Rhyme     | Rhyme scheme (ABAB etc.) preserved             | exact     |
| Timing    | Estimated sing time fits Whisper segment       | ±15%      |

## Tuning

- `MAX_ATTEMPTS` in orchestrator.py — default 4, lower for speed
- `TOLERANCE` in syllable_critic.py — ±1 syllable
- `TOLERANCE_RATIO` in timing_critic.py — ±15% of segment duration
- `CHARS_PER_SECOND` env var — default 14.0, adjust for song tempo

## Connecting to your audio pipeline

Your Whisper output looks like:
```json
{
  "segments": [
    {"id": 0, "text": "Never gonna give you up", "start": 0.5, "end": 2.1},
    ...
  ]
}
```

POST that to `/translate` with a `target_lang` field.
Poll `/status/{job_id}` until `status == "done"`.
