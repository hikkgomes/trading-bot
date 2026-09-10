"""Persist book ownership without assigning or moving existing account resources."""

from alembic import op

revision = "platform_v2_0034"
down_revision = "platform_v2_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE strategy_book (
            id VARCHAR(80) PRIMARY KEY, product_id VARCHAR(80) NOT NULL,
            artefact_hash VARCHAR(80) NOT NULL REFERENCES strategy_artefact(id),
            strategy_version_id VARCHAR(160) NOT NULL REFERENCES strategy_version(id),
            execution_mode VARCHAR(40) NOT NULL, sleeve_id VARCHAR(160) NOT NULL,
            account_id VARCHAR(160) NOT NULL, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            CONSTRAINT uq_strategy_book_artefact_mode UNIQUE(artefact_hash, execution_mode, sleeve_id),
            CONSTRAINT ck_strategy_book_mode CHECK(execution_mode IN ('paper','live'))
        );
        CREATE INDEX ix_strategy_book_product_id ON strategy_book(product_id);
        CREATE INDEX ix_strategy_book_account_id ON strategy_book(account_id);
        CREATE TABLE book_event (
            id VARCHAR(160) PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE INDEX ix_book_event_book_time ON book_event ((payload->>'book_id'), created_at);
        CREATE TABLE book_fill_allocation (
            id VARCHAR(80) PRIMARY KEY, fill_id VARCHAR(240) NOT NULL REFERENCES fill(id),
            book_id VARCHAR(80) NOT NULL REFERENCES strategy_book(id),
            signed_quantity NUMERIC(38,18) NOT NULL, fee NUMERIC(38,18) NOT NULL,
            fee_asset VARCHAR(40) NOT NULL, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            CONSTRAINT uq_book_fill_allocation UNIQUE(fill_id,book_id),
            CONSTRAINT ck_book_fill_amounts CHECK(signed_quantity <> 0 AND fee >= 0)
        );
        CREATE INDEX ix_book_fill_allocation_fill_id ON book_fill_allocation(fill_id);
        CREATE INDEX ix_book_fill_allocation_book_id ON book_fill_allocation(book_id);
        CREATE TABLE account_portfolio_plan (
            id VARCHAR(80) PRIMARY KEY, product_id VARCHAR(80) NOT NULL,
            account_id VARCHAR(160) NOT NULL, decision_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            CONSTRAINT uq_account_decision_cutoff UNIQUE(product_id,account_id,decision_at)
        );
        CREATE TRIGGER strategy_book_append_only BEFORE UPDATE OR DELETE ON strategy_book
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER book_event_append_only BEFORE UPDATE OR DELETE ON book_event
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER book_fill_allocation_append_only BEFORE UPDATE OR DELETE ON book_fill_allocation
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER account_portfolio_plan_append_only BEFORE UPDATE OR DELETE ON account_portfolio_plan
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        REVOKE ALL ON strategy_book, book_event, book_fill_allocation, account_portfolio_plan
            FROM PUBLIC, trading_research, trading_evaluator, trading_agent;
        GRANT SELECT, INSERT ON strategy_book, book_event, book_fill_allocation, account_portfolio_plan
            TO trading_runtime;
        GRANT SELECT ON strategy_book, book_event, book_fill_allocation, account_portfolio_plan
            TO trading_operator;
        INSERT INTO evidence_invalidation
        SELECT 'paper-ownership:' || md5(id), CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'paper_assignment_ownership',
                'reason', 'legacy_shared_account_qualification_quarantined',
                'financial_history_preserved', true)
        FROM active_strategy_assignment
        WHERE execution_mode='paper' AND payload->>'book_id' IS NULL
        ON CONFLICT DO NOTHING;
    """)
    _conservation_constraints()


def _conservation_constraints() -> None:
    op.execute("""
        CREATE FUNCTION trading_platform_check_book_fill() RETURNS trigger
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
        DECLARE fid TEXT; original JSONB; intent JSONB; allocated NUMERIC; fees NUMERIC;
            expected NUMERIC; metadata JSONB; expected_fee NUMERIC;
        BEGIN
            IF TG_TABLE_NAME='fill' THEN fid := NEW.id; ELSE fid := NEW.fill_id; END IF;
            SELECT f.payload, o.payload INTO original, intent FROM public.fill f
                JOIN public.order_intent o ON o.id=f.order_id WHERE f.id=fid;
            metadata := COALESCE(intent->'metadata'->'target_metadata',intent->'metadata');
            SELECT sum(signed_quantity),sum(fee) INTO allocated,fees
                FROM public.book_fill_allocation WHERE fill_id=fid;
            IF allocated IS NULL AND metadata->>'book_id' IS NULL
                AND metadata->'book_contributions' IS NULL
                AND intent->'metadata'->'book_contributions' IS NULL THEN RETURN NEW; END IF;
            expected := (original->>'quantity')::NUMERIC *
                CASE WHEN original->>'side'='buy' THEN 1 ELSE -1 END;
            expected_fee := (original->>'fee')::NUMERIC;
            IF allocated IS DISTINCT FROM expected OR fees IS DISTINCT FROM expected_fee
                OR EXISTS (SELECT 1 FROM public.book_fill_allocation a WHERE a.fill_id=fid
                    AND (a.signed_quantity*expected<=0 OR
                         (original->>'fee_asset' IS NOT NULL AND a.fee_asset<>original->>'fee_asset'))) THEN
                RAISE EXCEPTION 'book allocations must conserve external fill quantity, side and fee';
            END IF;
            RETURN NEW;
        END $$;
        CREATE CONSTRAINT TRIGGER fill_book_conservation AFTER INSERT ON fill
            DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION trading_platform_check_book_fill();
        CREATE CONSTRAINT TRIGGER allocation_book_conservation AFTER INSERT ON book_fill_allocation
            DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION trading_platform_check_book_fill();
        REVOKE ALL ON FUNCTION trading_platform_check_book_fill() FROM PUBLIC;
    """)


def downgrade() -> None:
    raise RuntimeError("Book ownership and financial allocation history must be retained")
