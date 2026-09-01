# Daily research supervisor

Work from `/home/alfred/trading-bot`. This is a research review, not live
authorisation.

## Read

Use the local authenticated control API or the sanitised agent-review record to
inspect the latest PostgreSQL report. Review candidate age by state, dataset
availability, first rejection reasons, duplicate feedback, family coverage,
generation feedback, forward summaries, and unresolved operational alerts.

Never read credentials, exchange account details, protected holdout data,
private order data, or live approval records.

## Decide

Choose zero to three bounded actions only when evidence supports them:

- create a distinct typed thesis;
- revise or retry a named research lineage;
- request a bounded research test;
- retire an exhausted lineage.

Do not repeat an existing action. State the evidence, expected outcome, and a
measurable falsification rule. Preserve product and family scope. A
`btc_accumulation` thesis must use BTCUSDT spot.

## Submit

Submit an `openclaw.agent_proposal/v1` payload to `POST /agent/proposals` via
the configured untrusted integration. The payload may contain a typed economic
thesis, bounded research requests, and research-only code proposals. It must
not contain results, approval flags, credentials, live instructions, order
requests, protected data, or risk decisions.

The trusted platform validates the proposal, assigns the normal PostgreSQL
dataset plan, deduplicates and lineage-links it, and sends it through the same
research evaluator as every other candidate. OpenClaw cannot grant authority
or submit orders.

Record the review in the platform agent-review store, including a no-op review.
Notify Henrique only for a material research conclusion, a service or risk
problem, or a decision that needs explicit authorisation. Otherwise return
`NO_REPLY`.
