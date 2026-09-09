-- Synthetic legacy exposure fixture; no customer data.
CREATE TABLE memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
INSERT INTO "memory_meta" VALUES ('schema_version','1');
INSERT INTO "memory_meta" VALUES ('format','autonomous_strategy_experiment_memory');
INSERT INTO "memory_meta" VALUES ('evaluation_engine_scopes_backfill_v1','complete');
INSERT INTO "memory_meta" VALUES ('holdout_cohorts_backfill_v1','complete');
INSERT INTO "memory_meta" VALUES ('protected_intervals_backfill_v1','complete');
CREATE TABLE strategies (
                    behavior_hash TEXT PRIMARY KEY,
                    canonical_spec_json TEXT NOT NULL,
                    primary_spec_json TEXT NOT NULL,
                    primary_strategy_id TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    primitive_tokens_json TEXT NOT NULL,
                    novelty_tokens_json TEXT NOT NULL,
                    novelty_score REAL NOT NULL,
                    nearest_behavior_hash TEXT,
                    product TEXT,
                    opportunity_type TEXT,
                    created_at TEXT NOT NULL,
                    holdout_exposed_at TEXT,
                    retired_at TEXT,
                    retirement_reason TEXT,
                    FOREIGN KEY(nearest_behavior_hash) REFERENCES strategies(behavior_hash)
                );
INSERT INTO "strategies" VALUES ('sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b','{"direction":"long","exit":{"horizon_bars":24,"stop_loss":0.01,"take_profit":0.02},"product":"active_income","trigger":[{"feature":"close","op":"gt","reference":100,"timeframe":"1h"}]}','{"direction":"long","exit":{"horizon_bars":24,"stop_loss":0.01,"take_profit":0.02},"family":"momentum","id":"fixture","product":"active_income","trigger":[{"feature":"close","op":"gt","reference":100,"timeframe":"1h"}]}','fixture','seed','{"family":"momentum","product":"active_income"}','["feature:close","op:gt"]','["$.direction","$.direction=\"long\"","$.exit","$.exit.horizon_bars","$.exit.horizon_bars=24","$.exit.stop_loss","$.exit.stop_loss=0.01","$.exit.take_profit","$.exit.take_profit=0.02","$.product","$.product=\"active_income\"","$.trigger","$.trigger[]","$.trigger[].feature","$.trigger[].feature=\"close\"","$.trigger[].op","$.trigger[].op=\"gt\"","$.trigger[].reference","$.trigger[].reference=100","$.trigger[].timeframe","$.trigger[].timeframe=\"1h\""]',1.0,NULL,'active_income',NULL,'2026-09-09T01:28:05+00:00','2026-09-09T01:28:05+00:00',NULL,NULL);
CREATE TABLE strategy_identities (
                    strategy_id TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    submitted_spec_json TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    parent_hashes_json TEXT NOT NULL,
                    is_duplicate INTEGER NOT NULL CHECK(is_duplicate IN (0, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
INSERT INTO "strategy_identities" VALUES ('fixture','sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b','{"direction":"long","exit":{"horizon_bars":24,"stop_loss":0.01,"take_profit":0.02},"family":"momentum","id":"fixture","product":"active_income","trigger":[{"feature":"close","op":"gt","reference":100,"timeframe":"1h"}]}','seed','{"family":"momentum","product":"active_income"}','[]',0,'2026-09-09T01:28:05+00:00');
CREATE TABLE lineage_edges (
                    child_hash TEXT NOT NULL,
                    parent_hash TEXT NOT NULL,
                    parent_ordinal INTEGER NOT NULL,
                    generation_method TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(child_hash, parent_hash),
                    FOREIGN KEY(child_hash) REFERENCES strategies(behavior_hash),
                    FOREIGN KEY(parent_hash) REFERENCES strategies(behavior_hash),
                    CHECK(child_hash <> parent_hash)
                );
CREATE TABLE evaluations (
                    evaluation_key TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    dataset_json TEXT NOT NULL,
                    window_json TEXT NOT NULL,
                    protocol_json TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('claimed', 'completed')),
                    claimed_at TEXT NOT NULL,
                    completed_at TEXT,
                    outcome TEXT,
                    rejection_reasons_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
INSERT INTO "evaluations" VALUES ('sha256:6e152b4bc9f545886a81eb14205171e28752b71be3f8ce31864f1d777d1cdbc2','sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b','legacy-fixture','{"market":"futures","snapshot_id":"legacy-fixture","symbol":"BTCUSDT"}','{"end":"2025-07-01","start":"2025-06-01"}','{"fees_bps":10}','sha256:238c9d34455bbbaffaae6d8b2bcadd90e951d9524fad8399538516b965322b3e','holdout','claimed','2026-09-09T01:28:05+00:00',NULL,NULL,'[]','{}','{}');
CREATE TABLE holdout_claim_scopes (
                    lineage_root_hash TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    evaluation_key TEXT NOT NULL,
                    behavior_hash TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    PRIMARY KEY(lineage_root_hash, data_snapshot_id),
                    FOREIGN KEY(lineage_root_hash) REFERENCES strategies(behavior_hash),
                    FOREIGN KEY(evaluation_key) REFERENCES evaluations(evaluation_key),
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
INSERT INTO "holdout_claim_scopes" VALUES ('sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b','legacy-fixture','sha256:6e152b4bc9f545886a81eb14205171e28752b71be3f8ce31864f1d777d1cdbc2','sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b','2026-09-09T01:28:05+00:00');
CREATE TABLE holdout_cohorts (
                    scope_key TEXT PRIMARY KEY,
                    data_snapshot_id TEXT NOT NULL,
                    protocol_hashes_json TEXT NOT NULL,
                    member_hashes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
INSERT INTO "holdout_cohorts" VALUES ('sha256:747aaa43feab4dd09cebf4945e385cce0149d832906b7997db4f079527dd1ffe','legacy-fixture','["sha256:238c9d34455bbbaffaae6d8b2bcadd90e951d9524fad8399538516b965322b3e"]','["sha256:6fdfcf22d707510e3c79d25cf944818bfb3eeeb20ac22b3efe06a57e54bb051b"]','2026-09-09T01:28:05+00:00');
CREATE TABLE protected_intervals (
                    interval_key TEXT PRIMARY KEY,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_utc TEXT NOT NULL,
                    end_utc TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    dataset_evidence_json TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    cohort_scope_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(cohort_scope_key) REFERENCES holdout_cohorts(scope_key)
                );
INSERT INTO "protected_intervals" VALUES ('sha256:d373897a6ea3aec98cd80444a7022d4e6a8999db60715c954fb915d369007341','futures','BTCUSDT','2025-06-01T00:00:00.000000+00:00','2025-07-01T00:00:00.000000+00:00','legacy-fixture','{"market":"futures","snapshot_id":"legacy-fixture","symbol":"BTCUSDT"}','sha256:ad2758d4750804b9adeff92e1de8da03d84a10b46f0f2bf52c0cfb4e493cdcf9','sha256:747aaa43feab4dd09cebf4945e385cce0149d832906b7997db4f079527dd1ffe','2026-09-09T01:28:05+00:00');
CREATE TABLE evaluation_engine_scopes (
                    evaluation_key TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    research_engine_digest TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    FOREIGN KEY(evaluation_key) REFERENCES evaluations(evaluation_key)
                        ON DELETE CASCADE,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
