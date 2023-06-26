import os
import pathlib
from typing import Any

import pytest
import yaml

from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config

config = Config()

"""
Test cases for the AIConfig class, which handles loads the AI configuration
settings from a YAML file.
"""


@pytest.fixture(autouse=True)
def setup(tmp_path: pathlib.Path) -> Any:
    config.ai_settings_filepath = tmp_path / "ai_settings.yaml"
    config.workspace_path = tmp_path / "auto_gpt_workspace"
    (config.workspace_path).mkdir(parents=True, exist_ok=True)
    config.plugins_allowlist = ["plugin1", "plugin2", "plugin3"]
    yield

    if config.ai_settings_filepath.exists():
        config.ai_settings_filepath.unlink()


def test_old_config_reformat() -> None:
    """Test if an old config format is updated to the new one."""

    # Old format of the configuration
    old_yaml_content = """
    ai_name: McFamished
    ai_role: A hungry AI
    ai_goals:
    - 'Goal 1: Make a sandwich'
    - 'Goal 2, Eat the sandwich'
    - 'Goal 3 - Go to sleep'
    - 'Goal 4: Wake up'
    api_budget: 0.0
    """

    config_file = config.ai_settings_filepath
    config_file.write_text(old_yaml_content)

    all_configs, message = AIConfig.update_old_config(str(config_file))

    # Expected format after update
    expected_yaml_content = """
    configs:
      McFamished:
        ai_name: McFamished
        ai_role: A hungry AI
        ai_goals:
        - 'Goal 1: Make a sandwich'
        - 'Goal 2, Eat the sandwich'
        - 'Goal 3 - Go to sleep'
        - 'Goal 4: Wake up'
        api_budget: 0.0
        plugins: []
    """

    expected_all_configs = yaml.safe_load(expected_yaml_content)

    assert all_configs == expected_all_configs
    assert message == "successfully transformed to support multiple configurations."


def test_old_config_reformat_without_ai_name() -> None:
    """Test if an old config format is updated to the new one."""

    # Old format of the configuration
    old_yaml_content = """
    ai_role: A hungry AI
    ai_goals:
    - 'Goal 1: Make a sandwich'
    - 'Goal 2, Eat the sandwich'
    - 'Goal 3 - Go to sleep'
    - 'Goal 4: Wake up'
    api_budget: 0.0
    """

    config_file = config.ai_settings_filepath
    config_file.write_text(old_yaml_content)

    all_configs, message = AIConfig.update_old_config(str(config_file))

    # Expected format after update
    expected_yaml_content = """
    configs: {}
    """

    expected_all_configs = yaml.safe_load(expected_yaml_content)

    assert all_configs == expected_all_configs
    assert message == "no configuration(s) detected."


def test_goals_are_always_lists_of_strings() -> None:
    """Test if the goals attribute is always a list of strings."""

    yaml_content = """configs:
      McFamished:
        ai_goals:
        - 'Goal 1: Make a sandwich'
        - 'Goal 2, Eat the sandwich'
        - 'Goal 3 - Go to sleep'
        - 'Goal 4: Wake up'
        ai_role: A hungry AI
        api_budget: 0.0
    """

    config_file = config.ai_settings_filepath
    config_file.write_text(yaml_content)

    all_configs, message = AIConfig.load_all(str(config_file))

    ai_config, message = AIConfig.load("McFamished", str(config_file))

    assert ai_config is not None, "ai_config is None"
    assert len(ai_config.ai_goals) == 4
    assert ai_config.ai_goals[0] == "Goal 1: Make a sandwich"
    assert ai_config.ai_goals[1] == "Goal 2, Eat the sandwich"
    assert ai_config.ai_goals[2] == "Goal 3 - Go to sleep"
    assert ai_config.ai_goals[3] == "Goal 4: Wake up"

    config_file.write_text("")
    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())

    expected_yaml = yaml.safe_load(
        """configs:
      McFamished:
        ai_goals:
        - 'Goal 1: Make a sandwich'
        - 'Goal 2, Eat the sandwich'
        - 'Goal 3 - Go to sleep'
        - 'Goal 4: Wake up'
        ai_role: A hungry AI
        api_budget: 0.0
        plugins: []
    """
    )

    assert saved_yaml == expected_yaml


def test_ai_config_file_not_exists() -> None:
    """Test if file does not exist."""

    config_file = config.ai_settings_filepath

    ai_config, message = AIConfig.load("Test", str(config_file))
    assert ai_config is None


def test_ai_config_file_is_empty() -> None:
    """Test if file does not exist."""

    config_file = config.ai_settings_filepath
    config_file.write_text("")

    ai_config, message = AIConfig.load("Test", str(config_file))
    assert ai_config is None


def test_delete_method() -> None:
    """Test if the delete method properly removes an AI configuration from the file."""

    yaml_content = """configs:
      AI1:
        ai_goals:
        - Goal 1
        ai_role: Test role
        api_budget: 0.0
      AI2:
        ai_goals:
        - Goal 2
        ai_role: Another role
        api_budget: 0.0
    """

    config_file = config.ai_settings_filepath
    config_file.write_text(yaml_content)

    AIConfig().delete(
        config_file,
        "AI1",
    )

    ai_config, message = AIConfig.load("AI1", str(config_file))
    assert ai_config is None

    ai_config2, message = AIConfig.load("AI2", str(config_file))
    assert ai_config2 is not None

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None
    ai_config2 = None


def test_special_character_config() -> None:
    yaml_content = """configs:
      SpécialAI:
        ai_goals:
        - 'Gôal 1: Mäke a sàndwich'
        ai_role: 'A hùngry AI'
        api_budget: 0.0
    """

    config_file = config.ai_settings_filepath
    config_file.write_text(yaml_content, encoding="utf-8")

    ai_config, message = AIConfig.load("SpécialAI", str(config_file))

    assert ai_config.ai_goals == ["Gôal 1: Mäke a sàndwich"]
    assert ai_config.ai_role == "A hùngry AI"
    assert ai_config.api_budget == 0.0

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None


def test_handling_special_characters_configuration() -> None:
    config_file = config.ai_settings_filepath
    config_file.write_text(
        "configs:\n  AI1:\n    ai_goals: ['Goal with special characters: !@#$%^&*()']\n"
    )

    ai_config, message = AIConfig.load("AI1", str(config_file))

    assert len(ai_config.ai_goals) == 1
    assert ai_config.ai_goals[0] == "Goal with special characters: !@#$%^&*()"

    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())
    expected_yaml = {
        "configs": {
            "AI1": {
                "ai_goals": ["Goal with special characters: !@#$%^&*()"],
            }
        }
    }
    assert (
        saved_yaml["configs"]["AI1"]["ai_goals"][0]
        == "Goal with special characters: !@#$%^&*()"
    )

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None


def test_loading_large_configuration() -> None:
    config_file = config.ai_settings_filepath

    # Create a large configuration with 100 AI entries
    config_content = "configs:\n"
    for i in range(100):
        config_content += f"  AI{i+1}:\n    ai_goals: ['Goal {i+1}']\n    ai_role: test role {i+1}\n    api_budget: 0.0\n    plugins: []\n"

    config_file.write_text(config_content)

    ai_config, message = AIConfig.load("AI50", str(config_file))

    assert ai_config.ai_name == "AI50"
    assert ai_config.ai_goals == ["Goal 50"]
    assert ai_config.api_budget == 0.0

    # Clean up
    config_file.unlink()
    ai_config = None


def test_saving_large_configuration() -> None:
    config_file = config.ai_settings_filepath
    ai_config = AIConfig("AI1", ai_goals=["Goal 1"])

    # Create a large configuration with 100 AI entries
    config_content = "configs:\n"
    for i in range(100):
        config_content += f"  AI{i+1}:\n    ai_goals: ['Goal {i+1}']\n    ai_role: test role {i+1}\n    api_budget: 0.0\n    plugins: []\n"

    config_file.write_text(config_content)

    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())
    expected_yaml = {
        "configs": {
            "AI1": {
                "ai_goals": ["Goal 1"],
                "ai_role": "",
                "api_budget": 0.0,
                "plugins": [],
            }
        }
    }

    assert saved_yaml == expected_yaml

    # Clean up
    config_file.unlink()
    ai_config = None


def test_save() -> None:
    ai_config = AIConfig(
        "test_name",
        "test_role",
        ["test_goal1", "test_goal2"],
        0.0,
        ["test_plugin1", "test_plugin2"],
    )

    config_file = config.ai_settings_filepath
    ai_config.save(config_file)

    with open(config_file, "r", encoding="utf-8") as file:
        saved_configs = yaml.safe_load(file)

    assert saved_configs["configs"]["test_name"] == {
        "ai_role": "test_role",
        "ai_goals": ["test_goal1", "test_goal2"],
        "api_budget": 0.0,
        "plugins": ["test_plugin1", "test_plugin2"],
    }


def test_save_empty_name() -> None:
    ai_config = AIConfig(
        "",
        "test_role",
        ["test_goal1", "test_goal2"],
        0.0,
        ["test_plugin1", "test_plugin2"],
    )

    config_file = config.ai_settings_filepath

    with pytest.raises(
        ValueError,
        match="The AI name cannot be empty. The configuration was not saved.",
    ):
        ai_config.save(config_file)

    assert not os.path.exists(config_file)


def test_save_with_old_ai_name() -> None:
    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    config_file = config.ai_settings_filepath
    ai_config.save(config_file)

    with open(config_file, "r", encoding="utf-8") as file:
        saved_configs = yaml.safe_load(file)
    assert "ai1" in saved_configs["configs"]

    new_ai_config = AIConfig("ai1", "role2", ["goal2"], 0.0, ["plugin2"])

    new_ai_config.save(config_file, old_ai_name="ai1")

    with open(config_file, "r", encoding="utf-8") as file:
        saved_configs = yaml.safe_load(file)
    assert "ai1" in saved_configs["configs"]
    assert saved_configs["configs"]["ai1"]["ai_role"] == "role2"

    os.remove(config_file)


def test_save_with_empty_file() -> None:
    config_file = config.ai_settings_filepath
    open(config_file, "a").close()

    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    ai_config.save(config_file)

    with open(config_file, "r", encoding="utf-8") as file:
        saved_configs = yaml.safe_load(file)
    assert "ai1" in saved_configs["configs"]

    os.remove(config_file)


def test_delete_no_ai_name() -> None:
    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    config_file = config.ai_settings_filepath
    ai_config.save(config_file)

    with pytest.raises(
        ValueError,
        match="No AI name provided. Please provide an AI name to delete its configuration.",
    ):
        ai_config.delete(config_file)

    os.remove(config_file)


def test_delete_no_config_file() -> None:
    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    with pytest.raises(ValueError, match="No configurations to delete."):
        ai_config.delete("non_existing_file.yaml", "ai1")


def test_delete_empty_config_file() -> None:
    config_file = config.ai_settings_filepath
    open(config_file, "a").close()

    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    with pytest.raises(ValueError, match="No configurations to delete."):
        ai_config.delete(config_file, "ai1")

    os.remove(config_file)


def test_delete_non_existing_ai_name() -> None:
    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    config_file = config.ai_settings_filepath
    ai_config.save(config_file)

    with pytest.raises(ValueError, match="No configuration found for AI 'ai2'."):
        ai_config.delete(config_file, "ai2")

    os.remove(config_file)


def test_delete_success() -> None:
    ai_config = AIConfig("ai1", "role1", ["goal1"], 0.0, ["plugin1"])

    config_file = config.ai_settings_filepath
    ai_config.save(config_file)

    ai_config.delete(config_file, "ai1")  # No need to capture a return value here

    with open(config_file, "r", encoding="utf-8") as file:
        saved_configs = yaml.safe_load(file)
    assert "ai1" not in saved_configs["configs"]
