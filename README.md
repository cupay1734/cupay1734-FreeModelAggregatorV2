# FreeModelAggregatorV2

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/cupay1734/cupay1734-FreeModelAggregatorV2/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/cupay1734/cupay1734-FreeModelAggregatorV2/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten free-tier
model providers — second iteration, tuned to spend as few upstream calls as possible.
It reads the providers' catalogs and routes every request to the highest-rated free
model that meets the request's requirements (tools, vision, JSON mode, context length)
and still has budget, speaking that provider's own wire (OpenAI, Anthropic or Gemini).

What is new in this iteration is failure-aware routing. The router tracks each
provider's refusal streak: a provider that has just refused is not asked again within
the same request, retries are re-ranked toward providers whose next call is safest
(a provider never yet called cannot be on its failure schedule), and a provider with
consecutive refusals is demoted — first by a rating penalty, then to last resort —
so an outage is probed a couple of times, not once per request. A success clears the
streak. In paired practice trials against the first iteration this serves the same
20/20 requests at the same quality while spending roughly two fewer upstream calls
per trial. Routing is a pure function of the requests and the answers already seen:
no wall clock, no randomness, so a replay walks the same routes.

## Run it

```sh
node main.ts          # serves; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
node main.ts --list   # prints the catalog the server registers, and exits
```

Node 22, standard library only.

---

<sub>Entered on [make.fun](https://make.fun/?ref=s522t60h&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
