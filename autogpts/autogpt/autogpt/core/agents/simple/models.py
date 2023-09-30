from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from autogpt.core.tools import ToolsRegistrySettings
from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
    BaseAgentSystemSettings,
)
from autogpt.core.planning import PlannerSettings
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings

if TYPE_CHECKING:
    pass


class SimpleAgentSystems(BaseAgentSystems):
    tool_registry: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation

    class Config(BaseAgentSystems.Config):
        pass


class SimpleAgentConfiguration(BaseAgentConfiguration):
    systems: SimpleAgentSystems
    agent_name: str = Field(default="New Agent")
    agent_role: Optional[str] = Field(default=None)
    agent_goals: Optional[list[str]] = Field(default=None)
    agent_goal_sentence: Optional[str] = Field(default=None)

    class Config(BaseAgentConfiguration.Config):
        pass


class SimpleAgentSystemSettings(BaseAgentSystemSettings):
    configuration: SimpleAgentConfiguration
    # user_id: Optional[uuid.UUID] = Field(default=None)
    # agent_id: Optional[uuid.UUID] = Field(default=None)

    class Config(BaseAgentSystemSettings.Config):
        pass


class SimpleAgentSettings(BaseAgentSettings):
    agent: SimpleAgentSystemSettings
    openai_provider: OpenAISettings
    tool_registry: ToolsRegistrySettings
    planning: PlannerSettings
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)
    agent_name: str = Field(default="New Agent")
    agent_role: Optional[str] = Field(default=None)
    agent_goals: Optional[list] = Field(default=None)
    agent_goal_sentence: Optional[list] = Field(default=None)
    agent_class: str = Field(default="autogpt.core.agents.simple.main.SimpleAgent")

    class Config(BaseAgentSettings.Config):
        pass