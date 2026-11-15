# Free-model aggregator 02-01

One server (Python 3.13, stdlib only) that aggregates make.fun's fourteen
free OpenRouter models behind an OpenAI-compatible and an
Anthropic-compatible wire, for competition `free-model-aggregator-02`.

## Strategy

**Capability-matrix routing with a refusal-aware, rating-ordered fallback
chain.** Every prompt is matched against the brief's model table (rating
plus tools / vision / JSON / context capabilities). The router serves a
prompt with the highest-rated model that meets its requirements. On a
refusal it walks the rating-ordered fallback chain: a 5xx is retried once
on the same model (transient), a 429 or a second failure demotes that
model for the rest of the trial and the next model in the chain is tried
(max 3 attempts per prompt, keeping the call count low for tie-breaks).
Every refusal is reported on the response in the
`x-makefun-upstream-failures` header so the failover column can see
recovered retries.

All choices are functions of the brief, the request and the gateway
answers seen so far: no wall clock and no unseeded randomness, so a replay
leaves the same state export byte for byte.

## Run

```
python3 main.py --list     # print the catalog and exit 0
python3 main.py            # register with the env, serve, exit 0 when done
```

The trial sets `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL`,
`MAKEFUN_TRIAL_SEED`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Every free
model is dialled on the gateway's Chat Completions route; Anthropic-shaped
requests are translated into that body and the answer translated back.
