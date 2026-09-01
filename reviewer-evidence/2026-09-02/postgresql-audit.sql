\pset pager off
\timing off

\echo '=== DATABASE AND MIGRATIONS ==='
SELECT current_database() AS database_name, now() AS observed_at;

SELECT version, applied_at, content_hash, revision_hash
FROM schema_migration
ORDER BY applied_at, version;

\echo '=== EXPERIMENTS BY STATE ==='
SELECT state, count(*) AS experiments
FROM experiment
GROUP BY state
ORDER BY state;

\echo '=== EXPERIMENTS BY DAY ==='
SELECT date_trunc('day', submitted_at) AS day,
       count(*) AS experiments
FROM experiment
GROUP BY 1
ORDER BY 1;

\echo '=== DEFINITIONS BY PRODUCT, SOURCE AND FAMILY ==='
SELECT product_id,
       source_type,
       COALESCE(definition->>'family', 'unknown') AS family,
       count(*) AS definitions
FROM strategy_definition
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

\echo '=== CANDIDATE IDENTITIES AND DUPLICATES ==='
SELECT is_duplicate,
       count(*) AS identities
FROM strategy_identity
GROUP BY is_duplicate
ORDER BY is_duplicate;

\echo '=== VALIDATION ATTRITION ==='
SELECT stage,
       accepted,
       COALESCE(reason_code, 'accepted') AS reason_code,
       count(*) AS decisions
FROM validation_stage
GROUP BY stage, accepted, COALESCE(reason_code, 'accepted')
ORDER BY stage, accepted, decisions DESC;

\echo '=== FIRST REJECTION PER EXPERIMENT ==='
WITH first_rejection AS (
    SELECT experiment_id,
           stage,
           reason_code,
           evaluated_at,
           row_number() OVER (
               PARTITION BY experiment_id
               ORDER BY evaluated_at, id
           ) AS ordinal
    FROM validation_stage
    WHERE accepted IS FALSE
)
SELECT stage,
       COALESCE(reason_code, 'unknown') AS reason_code,
       count(*) AS experiments
FROM first_rejection
WHERE ordinal = 1
GROUP BY stage, COALESCE(reason_code, 'unknown')
ORDER BY experiments DESC, stage, reason_code;

\echo '=== CANDIDATES NEVER EVALUATED ==='
SELECT e.provider,
       e.state,
       count(*) AS candidates,
       min(e.submitted_at) AS first_submitted,
       max(e.submitted_at) AS last_submitted
FROM experiment e
WHERE NOT EXISTS (
    SELECT 1
    FROM validation_stage s
    WHERE s.experiment_id = e.id
)
GROUP BY e.provider, e.state
ORDER BY candidates DESC, e.provider, e.state;

\echo '=== STAGE LIFECYCLE TIME ==='
SELECT count(*) AS experiments,
       min(first_stage_at) AS first_stage_at,
       max(last_stage_at) AS last_stage_at,
       min(elapsed) AS shortest_elapsed,
       avg(elapsed) AS average_elapsed,
       max(elapsed) AS longest_elapsed
FROM (
    SELECT experiment_id,
           min(evaluated_at) AS first_stage_at,
           max(evaluated_at) AS last_stage_at,
           max(evaluated_at) - min(evaluated_at) AS elapsed
    FROM validation_stage
    GROUP BY experiment_id
) lifecycle;

\echo '=== STAGE LIFECYCLE BY STAGE COUNT ==='
SELECT stage_records,
       count(*) AS experiments
FROM (
    SELECT experiment_id,
           count(*) AS stage_records
    FROM validation_stage
    GROUP BY experiment_id
) lifecycle
GROUP BY stage_records
ORDER BY stage_records;

\echo '=== THESIS TRIAL BUDGET USE ==='
SELECT creator_identity,
       count(*) AS theses,
       sum(cumulative_trial_budget) AS trial_budget,
       sum(claimed_trials) AS claimed_trials,
       sum(remaining_trials) AS remaining_trials,
       min(remaining_trials) AS minimum_remaining_trials,
       max(remaining_trials) AS maximum_remaining_trials
FROM (
    SELECT rt.id,
           rt.creator_identity,
           rt.cumulative_trial_budget,
           count(tt.id) AS claimed_trials,
           rt.cumulative_trial_budget - count(tt.id) AS remaining_trials
    FROM research_thesis rt
    LEFT JOIN thesis_trial tt
           ON tt.thesis_id = rt.id
    GROUP BY rt.id, rt.creator_identity, rt.cumulative_trial_budget
) budgets
GROUP BY creator_identity
ORDER BY creator_identity;

\echo '=== DATASETS BY PRODUCT AND ROLE ==='
SELECT payload->>'product_id' AS product_id,
       COALESCE(payload->>'role', 'unspecified') AS role,
       COALESCE(payload->'payload'->>'diagnostic', 'false') AS diagnostic,
       COALESCE(payload->'payload'->>'synthetic', 'false') AS synthetic,
       count(*) AS snapshots,
       min(created_at) AS first_created,
       max(created_at) AS last_created
FROM dataset_snapshot
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4;

\echo '=== HOLDOUT CLAIMS AND OUTCOMES ==='
SELECT
    (SELECT count(*) FROM holdout_claim) AS claims,
    (SELECT count(*) FROM holdout_outcome) AS outcomes,
    (SELECT count(*) FROM holdout_outcome WHERE accepted IS TRUE) AS accepted_outcomes;

\echo '=== ARTEFACTS ==='
SELECT payload->>'product_id' AS product_id,
       payload->>'strategy_version_id' AS strategy_version_id,
       count(*) AS artefacts,
       min(created_at) AS first_created,
       max(created_at) AS last_created
FROM strategy_artefact
GROUP BY 1, 2
ORDER BY 1, 2;

\echo '=== ACTIVE ASSIGNMENTS ==='
SELECT product_id,
       lifecycle_state,
       execution_mode,
       active,
       count(*) AS assignments,
       count(instrument_id) AS instrument_scoped,
       count(universe_id) AS universe_scoped,
       min(assigned_at) AS first_assigned,
       max(assigned_at) AS last_assigned
FROM active_strategy_assignment
GROUP BY product_id, lifecycle_state, execution_mode, active
ORDER BY product_id, lifecycle_state, execution_mode, active;

\echo '=== FORWARD OBSERVATIONS ==='
SELECT product_id,
       strategy_version_id,
       count(*) AS observations,
       min(observed_at) AS first_observation,
       max(observed_at) AS last_observation,
       count(*) FILTER (
           WHERE COALESCE(
               payload->'observation'->>'accepted',
               payload->>'accepted'
           ) = 'true'
       ) AS accepted_rows,
       count(*) FILTER (
           WHERE COALESCE(
               payload->'observation'->>'accepted',
               payload->>'accepted'
           ) = 'false'
       ) AS rejected_rows
FROM forward_paper_observation
GROUP BY product_id, strategy_version_id
ORDER BY product_id, strategy_version_id;

\echo '=== FORWARD REASONS ==='
SELECT COALESCE(
           payload->'observation'->>'reason_code',
           payload->>'reason_code',
           'missing'
       ) AS reason_code,
       count(*) AS observations
FROM forward_paper_observation
GROUP BY 1
ORDER BY observations DESC, reason_code;

\echo '=== PROMOTION EVENTS ==='
SELECT payload->>'prior_state' AS prior_state,
       payload->>'next_state' AS next_state,
       payload->>'accepted' AS accepted,
       payload->>'reason_code' AS reason_code,
       count(*) AS events
FROM promotion_event
GROUP BY 1, 2, 3, 4
ORDER BY events DESC, prior_state, next_state;

\echo '=== APPROVALS ==='
SELECT product_id,
       status,
       count(*) AS approvals,
       min(approved_at) AS first_approval,
       max(approved_at) AS last_approval
FROM strategy_approval
GROUP BY product_id, status
ORDER BY product_id, status;

\echo '=== PREFLIGHTS ==='
SELECT product_id,
       accepted,
       count(*) AS preflights,
       min(checked_at) AS first_checked,
       max(checked_at) AS last_checked
FROM production_preflight
GROUP BY product_id, accepted
ORDER BY product_id, accepted;

\echo '=== JOB QUEUE ==='
SELECT name,
       state,
       count(*) AS jobs,
       max(attempts) AS maximum_attempts,
       min(available_at) AS oldest_available,
       max(available_at) AS newest_available
FROM job
GROUP BY name, state
ORDER BY name, state;

\echo '=== JOB ATTEMPT FAILURES ==='
SELECT split_part(COALESCE(error, 'missing'), ':', 1) AS error_type,
       count(*) AS failures,
       min(started_at) AS first_failure,
       max(started_at) AS last_failure
FROM job_attempt
WHERE status NOT IN ('completed', 'succeeded', 'success')
GROUP BY 1
ORDER BY failures DESC, error_type;

\echo '=== SCHEDULES ==='
SELECT id,
       job_name,
       state,
       interval_seconds,
       last_run_at,
       next_run_at,
       now() - last_run_at AS age_since_last_run,
       last_job_id
FROM platform_schedule
ORDER BY job_name;

\echo '=== WORKERS ==='
SELECT node_id,
       role,
       status,
       count(*) AS workers,
       min(last_heartbeat) AS oldest_heartbeat,
       max(last_heartbeat) AS newest_heartbeat
FROM worker
GROUP BY node_id, role, status
ORDER BY node_id, role, status;

\echo '=== SERVICE HEARTBEATS ==='
SELECT DISTINCT ON (service_name, node_id)
       service_name,
       node_id,
       healthy,
       observed_at,
       now() - observed_at AS heartbeat_age
FROM service_heartbeat
ORDER BY service_name, node_id, observed_at DESC;

\echo '=== UNIVERSE ATTRITION ==='
SELECT us.universe_id,
       um.eligible,
       COALESCE(um.reason_code, 'missing') AS reason_code,
       count(*) AS instruments
FROM universe_member um
JOIN universe_snapshot us
  ON us.id = um.snapshot_id
GROUP BY us.universe_id, um.eligible, COALESCE(um.reason_code, 'missing')
ORDER BY us.universe_id, um.eligible DESC, instruments DESC;
