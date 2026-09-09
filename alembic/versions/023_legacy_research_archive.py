"""Preserve legacy protected exposure and strategy disposition records."""

from alembic import op

revision = "platform_v2_0023"
down_revision = "platform_v2_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE legacy_research_archive (
            id VARCHAR(160) PRIMARY KEY,
            source_hash VARCHAR(80) NOT NULL UNIQUE,
            content_hash VARCHAR(80) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE TRIGGER legacy_research_archive_append_only BEFORE UPDATE OR DELETE ON legacy_research_archive
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        REVOKE ALL ON legacy_research_archive FROM trading_runtime;
        GRANT SELECT ON legacy_research_archive TO trading_runtime;
        REVOKE ALL ON legacy_research_archive FROM trading_research, trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Protected exposure archive downgrade is forbidden")
