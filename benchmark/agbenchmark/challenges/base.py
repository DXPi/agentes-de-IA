import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, ClassVar, Optional

import pytest
from agent_protocol_client import AgentApi, Step
from colorama import Fore, Style
from pydantic import BaseModel, Field

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, DifficultyLevel, EvalResult

logger = logging.getLogger(__name__)


class ChallengeInfo(BaseModel):
    eval_id: str = ""
    name: str
    task: str
    task_artifacts_dir: Optional[Path] = None
    category: list[Category]
    difficulty: Optional[DifficultyLevel] = None
    description: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    reference_answer: Optional[str]

    source_uri: str
    """Internal reference indicating the source of the challenge specification"""


class BaseChallenge(ABC):
    """
    The base class and shared interface for all specific challenge implementations.
    """

    info: ClassVar[ChallengeInfo]

    @classmethod
    @abstractmethod
    def from_source_uri(cls, source_uri: str) -> type["BaseChallenge"]:
        """
        Construct an individual challenge subclass from a suitable `source_uri` (as in
        `ChallengeInfo.source_uri`).
        """
        ...

    @abstractmethod
    def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
    ) -> None:
        """
        Test method for use by Pytest-based benchmark sessions. Should return normally
        if the challenge passes, and raise a (preferably descriptive) error otherwise.
        """
        ...

    @classmethod
    async def run_challenge(
        cls, config: AgentBenchmarkConfig, timeout: int, *, mock: bool = False
    ) -> AsyncIterator[Step]:
        """
        Runs the challenge on the subject agent with the specified timeout.
        Also prints basic challenge and status info to STDOUT.

        Params:
            config: The subject agent's benchmark config.
            timeout: Timeout (seconds) after which to stop the run if not finished.

        Yields:
            Step: The steps generated by the agent for the challenge task.
        """
        # avoid circular import
        from agbenchmark.agent_api_interface import run_api_agent

        print()
        print(
            f"{Fore.MAGENTA + Style.BRIGHT}{'='*24} "
            f"Starting {cls.info.name} challenge"
            f" {'='*24}{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}Timeout:{Fore.RESET} {timeout} seconds")
        print(f"{Fore.CYAN}Task:{Fore.RESET} {cls.info.task}")

        print()
        logger.debug(f"Starting {cls.info.name} challenge run")
        i = 0
        async for step in run_api_agent(
            cls.info.task, config, timeout, cls.info.task_artifacts_dir, mock=mock
        ):
            i += 1
            print(f"[{cls.info.name}] - step {step.name} ({i}. request)")
            yield step
        logger.debug(f"Finished {cls.info.name} challenge run")

    @classmethod
    @abstractmethod
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    ) -> list[EvalResult]:
        ...
