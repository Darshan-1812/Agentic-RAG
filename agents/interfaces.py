"""Abstract interfaces for Member 3: Agentic Workflow & FastMCP Lead."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from core.schemas import QueryRequest, InsufficientEvidenceResponse


class BaseLangGraphSupervisor(ABC):
    """Interface for Tier 3 LangGraph Multi-Agent Supervisor."""

    @abstractmethod
    async def run(
        self,
        request: QueryRequest,
        initial_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes multi-hop cyclic graph reasoning with tool calls and evidence collection."""
        pass


class BaseMCPConnector(ABC):
    """Interface for FastMCP client calling tool endpoints (Postgres ERP, GitHub)."""

    @abstractmethod
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invokes a connected MCP tool."""
        pass
