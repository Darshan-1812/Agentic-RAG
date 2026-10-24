# FastMCP Servers (Member 3: Agentic Workflow & FastMCP Lead)

## Servers:
1. **PostgreSQL ERP Server (`mcp_servers/postgres_erp/`)**:
   - Exposes tools to query live ERP tables (e.g. shipment status, batch defect records, supplier delivery dates).
   - Read-only connection string security.
2. **GitHub Issues Server (`mcp_servers/github_issues/`)**:
   - Exposes tools to query open engineering blockers, pull requests, issue assignees, and repository commit statuses.
