SELECT 'strategies' AS section, count(*) AS count FROM strategies;
SELECT 'strategy_identities' AS section, count(*) AS count FROM strategy_identities;
SELECT 'evaluations' AS section, count(*) AS count FROM evaluations;
SELECT 'holdout_claim_scopes' AS section, count(*) AS count FROM holdout_claim_scopes;
SELECT 'holdout_cohorts' AS section, count(*) AS count FROM holdout_cohorts;
SELECT 'protected_intervals' AS section, count(*) AS count FROM protected_intervals;

SELECT
    COALESCE(product, 'unknown') AS product,
    COALESCE(opportunity_type, 'unknown') AS opportunity_type,
    count(*) AS strategies,
    sum(CASE WHEN holdout_exposed_at IS NOT NULL THEN 1 ELSE 0 END) AS holdout_exposed,
    sum(CASE WHEN retired_at IS NOT NULL THEN 1 ELSE 0 END) AS retired
FROM strategies
GROUP BY product, opportunity_type
ORDER BY product, opportunity_type;

SELECT
    generation_method,
    count(*) AS identities,
    sum(is_duplicate) AS duplicates
FROM strategy_identities
GROUP BY generation_method
ORDER BY identities DESC, generation_method;

SELECT
    phase,
    status,
    COALESCE(outcome, 'missing') AS outcome,
    count(*) AS evaluations,
    min(claimed_at) AS first_claimed,
    max(COALESCE(completed_at, claimed_at)) AS last_activity
FROM evaluations
GROUP BY phase, status, COALESCE(outcome, 'missing')
ORDER BY phase, status, outcome;

SELECT
    behavior_hash,
    count(*) AS evaluations,
    min(claimed_at) AS first_claimed,
    max(COALESCE(completed_at, claimed_at)) AS last_activity
FROM evaluations
GROUP BY behavior_hash
ORDER BY evaluations DESC
LIMIT 20;
