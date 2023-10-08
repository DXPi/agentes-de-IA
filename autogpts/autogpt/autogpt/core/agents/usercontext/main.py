from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, List, Tuple, Optional
from pydantic import Field

from autogpt.core.tools import ToolResult, SimpleToolRegistry
from autogpt.core.agents.base.main import BaseAgent
from autogpt.core.agents.usercontext.loop import UserContextLoop
from autogpt.core.agents.usercontext.models import (
    UserContextAgentConfiguration,
    UserContextAgentSettings,
    UserContextAgentSystems,
)
from autogpt.core.configuration import Configurable
from autogpt.core.memory.base import Memory
from autogpt.core.agents.simple.lib import SimplePlanner, Task, TaskStatusList
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import OpenAIProvider
from autogpt.core.workspace.simple import SimpleWorkspace

if TYPE_CHECKING:
    from autogpt.core.agents.base.loop import BaseLoopHook

from autogpt.core.agents.base.loop import BaseLoopHook


class UserContextAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_CONFIGURATION = UserContextAgentConfiguration
    CLASS_SETTINGS = UserContextAgentSettings
    CLASS_SYSTEMS = UserContextAgentSystems


    class SystemSettings(BaseAgent.SystemSettings):
        configuration : UserContextAgentConfiguration = UserContextAgentConfiguration()
        name="usercontext_agent"
        description="An agent that improve the quality of input provided by users."
        # user_id: Optional[uuid.UUID] = Field(default=None)
        # agent_id: Optional[uuid.UUID] = Field(default=None)

        class Config(BaseAgent.SystemSettings.Config):
            pass

    def __init__(
        self,
        settings: UserContextAgent.SystemSettings,
        logger: logging.Logger,
        memory: Memory,
        chat_model_provider: OpenAIProvider,
        workspace: SimpleWorkspace,
        planning: SimplePlanner,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ):
        super().__init__(
            settings=settings,
            logger=logger,
            memory=memory,
            workspace=workspace,
            user_id=user_id,
            agent_id=agent_id,
        )

        # These are specific
        self._chat_model_provider = chat_model_provider
        self._planning = planning

        self._loop = UserContextLoop(agent=self)

    def loophooks(self) -> UserContextLoop.LoophooksDict:
        if not self._loop._loophooks:
            self._loop._loophooks = {}
        return self._loop._loophooks

    def loop(self) -> UserContextLoop:
        return self._loop

    def add_hook(self, hook: BaseLoopHook, uuid: uuid.UUID):
        super().add_hook(hook, uuid)

    ################################################################################
    ################################ LOOP MANAGEMENT################################
    ################################################################################

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().start(
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def stop(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().stop(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    ################################################################################
    ################################FACTORY SPECIFIC################################
    ################################################################################

    @classmethod
    def _create_agent_custom_treatment(
        cls, agent_settings: UserContextAgentSettings, logger: logging.Logger
    ) -> None:
        pass

    @classmethod
    def _get_agent_from_settings(
        cls,
        agent_settings: UserContextAgentSettings,
        agent_args: list,
        logger: logging.Logger,
    ) -> Tuple[UserContextAgentSettings, list]:
        agent_args["chat_model_provider"] = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger,
        )
        from autogpt.core.agents.usercontext.strategies import (
            StrategiesSet,
            StrategiesSetConfiguration,
        )

        user_context_strategies = StrategiesSet.get_strategies(logger=logger)
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["chat_model_provider"]},
            strategies=user_context_strategies,
        )

        return agent_settings, agent_args

    """@classmethod
    def get_agent_from_settings(
        cls,
        agent_settings: UserContextAgentSettings,
        logger: logging.Logger,
    ) -> Agent:
        agent_settings, agent_args = super().get_agent_from_settings(
            agent_settings=agent_settings, 
            logger=logger
        )
        agent_args["chat_model_provider"] = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger,
        )
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["chat_model_provider"]},
        )

        # NOTE : Can't be moved to super() because require agent_args["chat_model_provider"]
        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["chat_model_provider"]},
        )
        return cls(**agent_args)"""

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: UserContextAgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")

        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.content

    def __repr__(self):
        return "UserContextAgent()"