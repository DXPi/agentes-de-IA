import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection
import pydantic

from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration import SystemConfiguration
from autogpt.core.configuration.schema import UserConfigurable
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import PluginLocation


class AbilityConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    location: PluginLocation
    packages_required: list[str] = UserConfigurable(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = UserConfigurable(default=False)
    workspace_required: bool = UserConfigurable(default=False)

    @pydantic.validator("location")
    def evaluate_location(cls, value: PluginLocation) -> PluginLocation:
        assert isinstance(value, PluginLocation)
        return value

    @pydantic.validator("packages_required")
    def evaluate_packages_required(cls, value: list) -> list:
        assert isinstance(value, list)
        for s in value:
            assert isinstance(s, str)
        return value

    @pydantic.validator("language_model_required")
    def evaluate_language_model_required(
        cls, value: LanguageModelConfiguration
    ) -> LanguageModelConfiguration:
        assert isinstance(value, LanguageModelConfiguration)
        return value

    @pydantic.validator("workspace_required")
    def evaluate_workspace_required(cls, value: bool) -> bool:
        assert isinstance(value, bool)
        return value

    @pydantic.validator("memory_provider_required")
    def evaluate_memory_provider_required(cls, value: bool) -> bool:
        assert isinstance(value, bool)
        return value


class Ability(abc.ABC):
    """A class representing an agent ability."""

    default_configuration: ClassVar[AbilityConfiguration]

    @classmethod
    def name(cls) -> str:
        """The name of the ability."""
        return inflection.underscore(cls.__name__)

    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """A detailed description of what the ability does."""
        ...

    @classmethod
    @abc.abstractmethod
    def arguments(cls) -> dict:
        """A dict of arguments in standard json schema format."""
        ...

    @classmethod
    def required_arguments(cls) -> list[str]:
        """A list of required arguments."""
        return []

    @abc.abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> AbilityResult:
        ...

    def __str__(self) -> str:
        return pformat(self.dump())

    def dump(self) -> dict:
        return {
            "name": self.name(),
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": self.arguments(),
                "required": self.required_arguments(),
            },
        }


class AbilityRegistry(abc.ABC):
    @abc.abstractmethod
    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ...

    @abc.abstractmethod
    def list_abilities(self) -> list[str]:
        ...

    @abc.abstractmethod
    def dump_abilities(self) -> list[dict]:
        ...

    @abc.abstractmethod
    def get_ability(self, ability_name: str) -> Ability:
        ...

    @abc.abstractmethod
    async def perform(self, ability_name: str, **kwargs: Any) -> AbilityResult:
        ...
