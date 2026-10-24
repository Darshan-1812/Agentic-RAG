"""Application configuration and SLA policy settings for ACR-RAG."""
from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SLAPolicyMode(str, Enum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    STRICT = "strict"


class SLAPolicyConfig(BaseModel):
    """SLA Governance policy parameters governing escalation & abstention."""
    mode: SLAPolicyMode
    tau_escalate: float = Field(..., description="Evidence threshold below which mid-flight escalation triggers")
    tau_abstain: float = Field(..., description="Evidence threshold below which formal abstention triggers")
    min_sentence_noul: float = Field(..., description="Required Noul entailment score per sentence in verification")
    description: str


DEFAULT_POLICY_PROFILES: Dict[SLAPolicyMode, SLAPolicyConfig] = {
    SLAPolicyMode.ECONOMY: SLAPolicyConfig(
        mode=SLAPolicyMode.ECONOMY,
        tau_escalate=0.35,
        tau_abstain=0.30,
        min_sentence_noul=0.60,
        description="Economy Mode: Minimizes token expenditure; escalates only when evidence is very weak (<0.35)."
    ),
    SLAPolicyMode.BALANCED: SLAPolicyConfig(
        mode=SLAPolicyMode.BALANCED,
        tau_escalate=0.65,
        tau_abstain=0.50,
        min_sentence_noul=0.75,
        description="Balanced Mode: Default enterprise profile; escalates when <0.65, abstains cleanly if <0.50."
    ),
    SLAPolicyMode.STRICT: SLAPolicyConfig(
        mode=SLAPolicyMode.STRICT,
        tau_escalate=0.85,
        tau_abstain=0.70,
        min_sentence_noul=0.88,
        description="Strict Zero-Hallucination Mode: High precision; requires Noul >= 0.88 on all claims or abstains."
    ),
}


class Settings(BaseSettings):
    # Jev Control Plane settings
    jev_api_key: str = "mock-jev-preview-key"
    jev_api_base_url: str = "https://api.typesafe.ai/v1"
    jev_price_per_million_input_tokens: float = 0.042
    
    # SLA Defaults
    default_sla_policy: SLAPolicyMode = SLAPolicyMode.BALANCED
    
    # Logging & env
    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_policy_config(self, mode: SLAPolicyMode = None) -> SLAPolicyConfig:
        target_mode = mode or self.default_sla_policy
        return DEFAULT_POLICY_PROFILES.get(target_mode, DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED])


settings = Settings()
