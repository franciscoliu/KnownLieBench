<h1 align="center">KnownLieBench</h1>

<h3 align="center">A Knowledge-Verified Benchmark of Emergent Deception in LLM Agents</h3>

<p align="center">
  <!-- arXiv badge (fill in the id):
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  -->
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/code-Apache--2.0-green.svg" alt="Code license">
  <img src="https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg" alt="Data license">
</p>

<p align="center">
  <a href="#">arXiv</a> |
  <a href="https://huggingface.co/datasets/franciscoliu/KnownLieBench">HuggingFace</a> |
  <a href="#quick-start-rocket">Quick Start</a> |
  <a href="#">Website</a> |
  <a href="#citation-star2">Citation</a>
</p>

---

## News :mega:

- **[08/2026]** 🎉 KnownLieBench is released!

---

## Overview :bulb:

Large language models are increasingly deployed as autonomous agents serving users on behalf of companies, placing them in settings where user and deployer interests can conflict. When an agent knows that a user is owed something its deployer would prefer to deny, does it remain honest? Answering this is difficult because false statements can reflect either ignorance or hallucination rather than deception. To address this challenge, we introduce **KnownLieBench**, a knowledge-verified benchmark that first confirms through a neutral probe that an agent knows a user's entitlement, and then tests whether it falsely denies that same entitlement once an incentive to do so is introduced. Specifically, KnownLieBench covers eight customer-service domains and 112 grounded cases, conducts multi-round dialogues with a trust-tracking customer agent, and separates deception emerging from incentive alone from deception produced under explicit instruction. Across eighteen proprietary and open-weight models, emergent deception varies substantially across model families and domains. We further use the benchmark as a post-training signal: honesty-directed fine-tuning reduces deception, while deception-graded fine-tuning increases lie success without significantly changing frequency. By separating lying from not knowing, KnownLieBench enables rigorous auditing and steering of agent honesty.

<p align="center">
  <img src="docs/assets/pipeline.png" width="100%" alt="KnownLieBench pipeline">
</p>

---

## The Benchmark :scales:

**8 domains, each pairing a fictional institution with a real governing rule:**

| Domain | Institution | Governing rule | Owed vs. company-favored |
|---|---|---|---|
| Retail refund | ShopRight | US retailer return practice | card refund vs. store credit |
| Airline refund | SkyJet Airlines | DOT 14 CFR Part 260 | original payment vs. voucher |
| Security deposit | Maple Grove Residential | Cal. Civ. Code §1950.5 | return vs. withhold |
| Vehicle recall | Atlas Motors | 49 U.S.C. §30120 | free remedy vs. deny |
| Subscription | StreamFlex | Cal. B&P Code §17602 | cancel vs. retain |
| Billing dispute | Meridian Card | Reg Z, 12 CFR 1026.13 | open dispute vs. deny |
| Insurance claim | Summit Health Plan | ERISA, 29 CFR 2560.503-1 | file claim vs. refuse |
| Debt collection | Sentinel Recovery Group | FDCPA & Reg F, 12 CFR 1006 | lawful vs. coercive handling |

**112 grounded cases** — 14 per domain: 8 *owed* cases where the customer is entitled to the contested outcome, and 6 *calibration* cases where no entitlement is at stake, so the honest answer — a truthful denial, or a truthful confirmation that the company's handling is lawful — is the correct one. Indiscriminate refusals and invented entitlements are caught in both directions.

**3 conditions**, identical environment, only the agent's private note changes:

- `none` — honest control: resolve the request accurately.
- `lie_incentive` — **emergent** deception: a private cost pressure against the owed outcome that never mentions lying. Any fabrication is the model's own.
- `instructed_lie` — **instructed** deception: an explicit private directive to deny with a plausible false reason.

**3 initial trust levels** (`high`, `neutral`, `low`) set how readily the simulated customer accepts claims versus verifies them against the policy; trust then evolves round to round based on what the agent does.

**Metrics** (computed by `scripts/score_dialogue.py`; the deception metrics use owed cells and gate-passed rounds only, while KPR is computed over all probed rounds):

- **KPR** — knowledge-pass rate: how often the neutral probe certifies the entitlement is known.
- **DR** (`lie_rate` in the panel) — deception rate: knowledge-verified false claims per scored round. With multiple domains scored, the panel also prints `macro_DR`, the unweighted mean over per-domain rates, which is the paper's aggregation.
- **DSR** (`DSR|lied`) — deception success rate: lies that were believed and cost the customer the owed outcome.
- **Detect** (`detect|lied`) — how often the customer caught the lie.

**Judge**: a fact-extraction judge (pinned `gpt-5.1-2025-11-13`) extracts per-round facts; verdicts are derived deterministically from those facts. Validated in the paper against 60 human-adjudicated labels (pooled κ = 0.815). Optional ensemble / decompose / verify wrapper graders are included for robustness checks.

---

## Installation :books:

```bash
git clone https://github.com/franciscoliu/KnownLieBench.git
cd KnownLieBench
pip install -e ".[dev]"          # add ".[local]" for the open-weight backend
cp .env.example .env             # then fill in the key(s) you plan to use
```

Requires Python 3.10+. The `[local]` extra pulls in `torch`, `transformers`, and `accelerate` for running open-weight agents on your own GPU.

---

## Quick Start :rocket:

**1. Zero-key demo** — one complete, scored episode with deterministic scripted players (no API calls, no GPU):

```bash
python scripts/run_benchmark.py --domain refund \
    --agent scripted_agent_deceptive --customer scripted_customer --grader scripted_grader \
    --conditions lie_incentive --levels high --n_traj 1 --rounds 2
python scripts/score_dialogue.py --dir outputs/dialogue --B 500
```

The scripted agent passes the knowledge gate, falsely claims the owed card refund is unavailable, and converts the customer to store credit — so the panel shows `lie_rate 1.00`, `DSR 1.00`, `detect 0.00`. Swap in `--agent scripted_agent_honest --conditions none` for the honest control (`lie_rate 0.00`). The whole test suite is offline too: `pytest -q` finishes in seconds with no keys.

**2. Evaluate an API model on one domain** (needs `OPENAI_API_KEY`):

```bash
python scripts/run_benchmark.py --domain airline --agent openai_agent_gpt4o_mini
python scripts/score_dialogue.py
```

Any OpenAI-compatible endpoint works the same way: fill in the `compat_agent` entry in `configs/models.yaml` (or export `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY`) and pass `--agent compat_agent`. This covers OpenRouter, self-hosted LiteLLM gateways, `vllm serve`, and vendor compatibility endpoints.

**3. Evaluate an open-weight model on your GPU** (needs `pip install -e ".[local]"`):

```bash
python scripts/run_benchmark.py --domain airline --agent local_llama31_8b
```

`model_path` in `configs/models.yaml` accepts a Hugging Face hub id (downloaded to your HF cache on first use) or a local snapshot directory. The three shipped configs are the paper's open-weight agents; an 8B model in bf16 fits a single 24 GB GPU.

---

## Reproducing the Paper's Protocol :straight_ruler:

The paper's panel runs every domain with the pinned apparatus (customer `gpt-4o-mini-2024-07-18`, judge `gpt-5.1-2025-11-13` — both need only `OPENAI_API_KEY`):

```bash
for DOM in refund airline deposit recall subscription billing insurance debt; do
  python scripts/run_benchmark.py --domain "$DOM" --agent <your_agent_cfg> --label mymodel_"$DOM"
done
python scripts/score_dialogue.py            # macro panel over all finished domains
```

Per model this is 3 conditions × 3 trust levels × 14 cases × 8 domains = **1,008 rounds**, and every round makes one gate call, several dialogue calls, and two judge calls. With the pinned judge this cost roughly **$12 per model** at 2026 API prices (measured from the `usage_summary` line each run writes) — set `--n_traj`/`--rounds` smaller, or run fewer domains in the loop, if you are experimenting.

Two reproducibility notes, matching the paper: the agent is sampled at temperature 0.7 (its behavioral distribution is the object of study) while the gate, customer, and judge run at 0; and no seed is set, so runs are statistically, not bit-wise, reproducible — the scorer reports bootstrap confidence intervals. `scripts/check_complete.py` guards against re-running finished domains; `scripts/regrade_domain.py` re-judges finished trajectories under a different grader config.

---

## Outputs :open_file_folder:

Each run writes to `outputs/dialogue/`:

- `trajectories_<label>_<domain>_<delta_set>.jsonl` — one row per round: the full transcript, gate result, judge facts, derived verdicts, trust trajectory, and terminal outcome; the first line records the resolved model/temperature provenance (`run_meta`), the last a token-usage summary.
- `summary_<label>_<domain>_<delta_set>.json` — per-trajectory aggregates.

`scripts/score_dialogue.py` folds every finished trajectory file in the directory into the panel: KPR decomposition, DR / DSR / Detect by condition, trust level, and case, with bootstrap CIs.

---

## Repository Structure :card_index_dividers:

```
configs/models.yaml            model configs (agents, apparatus, wrappers)
scripts/run_benchmark.py       the dialogue runner (start here)
scripts/score_dialogue.py      panel scoring + bootstrap CIs
scripts/agent_actions.py       action parsing shared by the runner
src/knownliebench/domains/     the 8 domain specs (prompts, gates, graders)
src/knownliebench/envs/        environments, tools, and the 112-case data
src/knownliebench/backends/    model providers (openai, openai_compatible, local_hf, scripted, mock)
                               plus the ensemble / decompose / verify grader wrappers
src/knownliebench/trust/       the trust-update models
tests/                         148 offline tests (no API keys needed)
```

---

## Citation :star2:

```bibtex
@article{knownliebench2026,
  title  = {Knowledge-Verified Emergent Deception in LLM Agents
            Under Conflicting Incentives},
  author = {Liu, Zheyuan and Zhao, Weiliang and Yuan, Xiangchi and Ma, Ningshan and Huang, Yue and Jiang, Meng},
  year   = {2026},
  note   = {Preprint}
}
```

## License :page_facing_up:

Code is released under [Apache-2.0](LICENSE); the case data under [CC BY 4.0](DATA_LICENSE). The statutes and regulations the cases cite are public domain; all institutions and people in the data are fictional.
