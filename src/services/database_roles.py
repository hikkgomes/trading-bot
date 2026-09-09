"""Reject service credentials that cross PostgreSQL security domains."""

from sqlalchemy import text

SERVICE_ROLES = frozenset(
    {
        "trading_runtime",
        "trading_research",
        "trading_agent",
        "trading_evaluator",
        "trading_operator",
        "trading_platform_owner",
    }
)


def check_connection_role(connection, required: str) -> None:
    if required not in SERVICE_ROLES:
        raise ValueError("unknown service database role")
    rows = (
        connection.execute(
            text("""
        SELECT rolname,
               pg_has_role(current_user, oid, 'MEMBER') AS effective_member,
               pg_has_role(session_user, oid, 'MEMBER') AS login_member,
               rolsuper, rolbypassrls, rolcreaterole,
               rolname IN (current_user, session_user) AS credential_role
        FROM pg_roles
    """)
        )
        .mappings()
        .all()
    )
    membership = {row["rolname"] for row in rows if row["effective_member"] or row["login_member"]}
    elevated = any(
        row["credential_role"] and (row["rolsuper"] or row["rolbypassrls"] or row["rolcreaterole"])
        for row in rows
    )
    forbidden = membership & (SERVICE_ROLES - {required})
    if required not in membership or forbidden or elevated:
        raise PermissionError(f"database credential must belong only to service domain {required}")


def check_service_roles(engine, config, services) -> None:
    domains = {
        domain
        for domain, members in config.security_domains.items()
        if set(services) & set(members)
    }
    if len(domains) != 1:
        raise PermissionError("one process cannot span database security domains")
    required = domains.pop().replace("-", "_")
    if required == "trading_agent":
        raise PermissionError("agent processes cannot open the platform database")
    with engine.connect() as connection:
        check_connection_role(connection, required)
