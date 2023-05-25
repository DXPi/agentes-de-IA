import pytest
from pytest_mock import MockerFixture

from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.challenges.utils import run_interaction_loop, run_multiple_times
from tests.utils import requires_api_key

CYCLE_COUNT = 3
from autogpt.agent import Agent


# @pytest.mark.skip("This challenge hasn't been beaten yet.") # Beaten on May 17 2023
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@run_multiple_times(3)
def test_information_retrieval_challenge_b(
    anthropic_information_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
) -> None:
    """
    Test the challenge_b function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    run_interaction_loop(monkeypatch, anthropic_information_agent, CYCLE_COUNT)

    file_path = str(anthropic_information_agent.workspace.get_path("output.txt"))
    content = read_file(file_path)
    assert "100" or "100k" in content, "Expected the file to contain 100"
