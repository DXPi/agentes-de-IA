from __future__ import annotations

import datetime
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Optional

import yaml
from pydantic import Field, root_validator

from AFAAS.configs import SystemSettings
from AFAAS.interfaces.agent.loop import BaseLoop  # Import only where it's needed
from AFAAS.lib.message_agent_agent import MessageAgentAgent
from AFAAS.lib.message_agent_llm import MessageAgentLLM
from AFAAS.lib.message_agent_user import MessageAgentUser
from AFAAS.lib.message_common import AFAASMessage, AFAASMessageStack
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class AbstractAgent(ABC):

    # _loophooks: Dict[str, BaseLoop.LoophooksDict] = {}
    _agent_type_: ClassVar[str] = __name__
    _agent_module_: ClassVar[str] = __module__ + "." + __name__
    class SystemSettings(SystemSettings):
        user_id: str
        modified_at: datetime.datetime = datetime.datetime.now()
        created_at: datetime.datetime = datetime.datetime.now()

        # @staticmethod
        # def _get_message_agent_user(agent_id):
        #     LOG.notice(f"Retriving : Agent - User Message history for {agent_id}")
        #     return []
        #     # return MessageAgentUser.get_from_db(agent_id)

        # @staticmethod
        # def _get_message_agent_agent(agent_id):
        #     LOG.notice(f"Retriving : Agent - Agent Message history for {agent_id}")
        #     return []
        #     # return MessageAgentAgent.get_from_db(agent_id)

        # @staticmethod
        # def _get_message_agent_llm(agent_id):
        #     LOG.notice(f"Retriving : Agent - LLM Message history for {agent_id}")
        #     return []
        #     # return MessageAgentLLM.get_from_db(agent_id)

        # Use lambda functions to pass the agent_id
        # message_agent_user: list[MessageAgentUser] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_user(self.agent_id)
        #     )
        # message_agent_agent: list[MessageAgentAgent] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_agent(self.agent_id)
        #     )
        # message_agent_llm: list[MessageAgentLLM] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_llm(self.agent_id)
        #     )


        # message_agent_user: list[MessageAgentUser] = []
        # message_agent_agent: list[MessageAgentAgent] = []
        # message_agent_llm: list[MessageAgentLLM] = []

        _message_agent_user: Optional[AFAASMessageStack] = Field(default=[])
        @property
        def message_agent_user(self) -> AFAASMessageStack:
            if self._message_agent_user is None:
                self._message_agent_user = AFAASMessageStack(
                    parent_task=self, description="message_agent_user"
                )
            return self._message_agent_user


        def __init__(self, **data):
            super().__init__(**data)
            for field_name, field_type in self.__annotations__.items():
                # Check if field_type is a class before calling issubclass
                if isinstance(field_type, type) and field_name in data and issubclass(field_type, AFAASMessageStack):
                    setattr(self, field_name, AFAASMessageStack(_stack=data[field_name]))

        # @root_validator(pre=True)
        # def set_default_messages(cls, values):
        #     agent_id = values.get("agent_id", "A" + str(uuid.uuid4()))
        #     values["message_agent_user"] = cls._get_message_agent_user(agent_id)
        #     values["message_agent_agent"] = cls._get_message_agent_agent(agent_id)
        #     values["message_agent_llm"] = cls._get_message_agent_llm(agent_id)
        #     return values


        class Config(SystemSettings.Config):
            AGENT_CLASS_FIELD_NAME : str = "_type_"
            AGENT_CLASS_MODULE_NAME : str = "_module_"

        @property
        def _type_(self):
            # == "".join(self.__class__.__qualname__.split(".")[:-1])  
            return self.__class__.__qualname__.split(".")[0]    

        @property
        def _module_(self):
            # Nested Class
            return self.__module__ + "." + self._type_

        @classmethod
        @property
        def settings_agent_class_(cls):
            return cls.__qualname__.partition(".")[0]

        @classmethod
        @property
        def settings_agent_module_(cls):
            return cls.__module__ + "." + ".".join(cls.__qualname__.split(".")[:-1])


AbstractAgent.SystemSettings.update_forward_refs()
