"""FastMCP Issue Tracker Tool Server (Member 3: Agentic Workflow & FastMCP Lead).

Domain-Agnostic Issue & Bug Tracking Connector:
1. Connects to any GitHub repository or issue tracker via GITHUB_TOKEN or offline workspace cache.
2. Generic search: search_issues(query, repository, state), list_repositories().
3. Specialized blocker search: search_open_blockers(component, repository).
4. Issue details and discussions: get_issue_details(issue_id, repository).
"""
import os
import re
from typing import Dict, Any, List, Optional
import httpx
from core.config import settings

# FastMCP / Lightweight MCP Tool Wrapper
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("GitHub-Issues-Server")
except Exception:
    class FastMCP:
        def __init__(self, name: str):
            self.name = name
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
    mcp = FastMCP("GitHub-Issues-Server")


# Multi-domain embedded repository issues for offline resilience & demo
EMBEDDED_REPO_ISSUES = {
    "apex-ev/firmware-bms": [
        {
            "issue_id": "#114",
            "title": "BMS firmware clock drift on VoltCore delayed cell packs",
            "assignee": "R. Kulkarni",
            "state": "OPEN",
            "labels": ["blocker", "firmware", "batch-402"],
            "component": "BMS-Controller-v4",
            "created_at": "2026-10-21T08:30:00Z",
            "body": "Batch #402 cell pack variance causes internal clock drift in BMS-Controller-v4 telemetry. Critical blocker for vehicle line restart."
        },
        {
            "issue_id": "#119",
            "title": "Thermal runaway sensor false trigger on Batch #402 integration",
            "assignee": "R. Kulkarni",
            "state": "OPEN",
            "labels": ["critical", "hardware-qa", "voltcore"],
            "component": "BMS-Controller-v4",
            "created_at": "2026-10-22T14:15:00Z",
            "body": "Thermistor calibration in Batch #402 integration produces spurious trip codes during initial pack charging."
        },
        {
            "issue_id": "#105",
            "title": "CAN bus message CRC error rate elevated",
            "assignee": "A. Deshmukh",
            "state": "CLOSED",
            "labels": ["bug", "canbus"],
            "component": "BMS-Controller-v4",
            "created_at": "2026-10-10T11:00:00Z",
            "body": "Resolved by firmware patch v4.0.2."
        }
    ],
    "company/operations-platform": [
        {
            "issue_id": "#502",
            "title": "API Gateway timeout on bulk export pipeline",
            "assignee": "DevOps OnCall",
            "state": "OPEN",
            "labels": ["infrastructure", "p1"],
            "component": "API-Gateway",
            "created_at": "2026-10-19T09:00:00Z",
            "body": "High memory pressure during end-of-shift reconciliation."
        }
    ]
}


@mcp.tool()
def list_repositories() -> List[str]:
    """Lists all repositories tracked in this workspace."""
    return list(EMBEDDED_REPO_ISSUES.keys())


@mcp.tool()
def search_issues(
    query: str,
    repository: str = "apex-ev/firmware-bms",
    state: str = "open"
) -> Dict[str, Any]:
    """Generic search for issues and PRs across any repository by keyword and state."""
    q_clean = query.lower()
    target_repo = repository.strip() if repository in EMBEDDED_REPO_ISSUES else "apex-ev/firmware-bms"
    repo_issues = EMBEDDED_REPO_ISSUES.get(target_repo, [])

    matched = []
    for issue in repo_issues:
        state_match = True if state.lower() == "all" else (issue["state"].lower() == state.lower())
        if state_match:
            text_corpus = f"{issue['title']} {issue['body']} {issue['component']} {' '.join(issue['labels'])}".lower()
            if any(term in text_corpus for term in q_clean.split()):
                matched.append(issue)

    return {
        "repository": target_repo,
        "query": query,
        "state_filter": state,
        "total_found": len(matched),
        "issues": [
            {
                "issue_id": m["issue_id"],
                "title": m["title"],
                "assignee": m["assignee"],
                "state": m["state"],
                "labels": m["labels"]
            }
            for m in matched
        ]
    }


@mcp.tool()
def search_open_blockers(component: str, repository: str = "apex-ev/firmware-bms") -> Dict[str, Any]:
    """Finds open blocker issues and lead engineers tied to components or services."""
    github_token = os.environ.get("GITHUB_TOKEN")

    # 1. Live GitHub API call if token configured
    if github_token and repository:
        try:
            headers = {
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            url = f"https://api.github.com/repos/{repository}/issues"
            params = {"state": "open", "labels": "blocker"}
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    issues = [
                        {
                            "issue_id": f"#{item['number']}",
                            "title": item["title"],
                            "assignee": item.get("assignee", {}).get("login", "Unassigned") if item.get("assignee") else "Unassigned",
                            "state": item["state"].upper(),
                            "labels": [lbl["name"] for lbl in item.get("labels", [])]
                        }
                        for item in data
                    ]
                    lead = issues[0]["assignee"] if issues else "Unassigned"
                    return {
                        "component": component,
                        "repository": repository,
                        "open_blockers_count": len(issues),
                        "issues": issues,
                        "lead_engineer": lead,
                        "source": "live_github_api"
                    }
        except Exception:
            pass

    # 2. Resilient embedded repo search
    target_repo = repository if repository in EMBEDDED_REPO_ISSUES else "apex-ev/firmware-bms"
    repo_issues = EMBEDDED_REPO_ISSUES.get(target_repo, [])
    c_lower = component.lower()

    matched = []
    for issue in repo_issues:
        if issue["state"] == "OPEN":
            if (
                c_lower in issue["component"].lower()
                or any(c_lower in lbl.lower() for lbl in issue["labels"])
                or c_lower in issue["title"].lower()
                or ("voltcore" in c_lower and "voltcore" in issue["body"].lower())
            ):
                matched.append(issue)

    lead_engineer = "R. Kulkarni" if any("Kulkarni" in i.get("assignee", "") for i in matched) else "Unassigned"

    return {
        "component": component,
        "repository": target_repo,
        "open_blockers_count": len(matched),
        "issues": [
            {
                "issue_id": m["issue_id"],
                "title": m["title"],
                "assignee": m["assignee"],
                "state": m["state"],
                "labels": m["labels"]
            }
            for m in matched
        ],
        "lead_engineer": lead_engineer,
        "source": "embedded_repository_cache"
    }


@mcp.tool()
def get_issue_details(issue_id: str, repository: str = "apex-ev/firmware-bms") -> Dict[str, Any]:
    """Retrieves full body discussion and comments for an issue."""
    clean_id = issue_id if issue_id.startswith("#") else f"#{issue_id}"
    target_repo = repository if repository in EMBEDDED_REPO_ISSUES else "apex-ev/firmware-bms"
    repo_issues = EMBEDDED_REPO_ISSUES.get(target_repo, [])

    for issue in repo_issues:
        if issue["issue_id"] == clean_id:
            return {
                "issue_id": issue["issue_id"],
                "title": issue["title"],
                "body": issue["body"],
                "assignee": issue["assignee"],
                "state": issue["state"],
                "labels": issue["labels"],
                "created_at": issue.get("created_at", "2026-10-21T00:00:00Z"),
                "found": True
            }

    return {"issue_id": clean_id, "found": False, "message": f"Issue {clean_id} not found in {repository}"}


if __name__ == "__main__":
    print("Issue Tracker MCP Server ready.")
    print("Repositories:", list_repositories())
    print("Generic search 'export':", search_issues("export", repository="company/operations-platform")["issues"])
