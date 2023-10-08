from __future__ import annotations

import uuid
import enum
import json
import platform
import re
from logging import Logger
from typing import TYPE_CHECKING, Callable, Optional


import distro

from autogpt.core.agents.base.agent_directives import BaseAgentDirectives

if TYPE_CHECKING:
    from autogpt.core.agents.simple import PlannerAgent


# prompting
from autogpt.core.prompting.base import LanguageModelClassification, RESPONSE_SCHEMA
from autogpt.core.prompting.planningstrategies import (
    PlanningPromptStrategiesConfiguration,
    PlanningPromptStrategy,
)

from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
    OpenAIProvider,
    ChatPrompt,
)

from autogpt.core.utils.json_schema import JSONSchema

from autogpt.core.prompting.utils.utils import (
    json_loads,
    to_numbered_list,
    to_string_list,
    indent,
)

from autogpt.core.agents.simple.lib.models.action_history import Episode


class ThinkStrategyFunctionNames(str, enum.Enum):
    THINK: str = "select_tool"


###
### CONFIGURATION
####
class ThinkStrategyConfiguration(PlanningPromptStrategiesConfiguration):
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.FAST_MODEL_16K
    )


###
### STRATEGY
####
class ThinkStrategy(PlanningPromptStrategy):
    default_configuration: ThinkStrategyConfiguration = ThinkStrategyConfiguration()
    STRATEGY_NAME = "select_tool"

    def __init__(
        self,
        logger: Logger,
        model_classification: LanguageModelClassification,
        **kwargs,
    ):
        super().__init__(
            logger=logger, model_classification=model_classification, **kwargs
        )

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent: "PlannerAgent",
        # instruction: str,
        **kwargs,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a select_tooling cycle
        """

        model_name = kwargs["model_name"]
        self._functions = agent._tool_registry.dump_tools()

        ###
        ### To Facilitate merge with AutoGPT changes
        ###
        event_history = False
        include_os_info = True
        del kwargs["tools"]
        tools = self._functions
        agent_directives = BaseAgentDirectives.from_file(agent=agent)
        extra_messages: list[ChatMessage] = []

        system_prompt = self._construct_system_prompt(
            agent=agent,
            agent_directives=agent_directives,
            tools=tools,
            include_os_info=include_os_info,
            **kwargs,
        )
        # system_prompt_tlength = count_message_tokens(ChatMessage.system(system_prompt))

        response_format_instr = self.response_format_instruction(
            agent=agent,
            model_name=model_name,
        )
        extra_messages.append(ChatMessage.system(response_format_instr))

        final_instruction_msg = ChatMessage.user(self._config.choose_action_instruction)
        # final_instruction_tlength = count_message_tokens(final_instruction_msg)

        if event_history:
            progress = self.compile_progress(
                event_history,
                # count_tokens=count_tokens,
                # max_tokens=(
                #     max_prompt_tokens
                #     - system_prompt_tlength
                #     - final_instruction_tlength
                #     - count_message_tokens(extra_messages)
                # ),
            )
            extra_messages.insert(
                0,
                ChatMessage.system(f"## Progress\n\n{progress}"),
            )
            pass

        messages = [
            ChatMessage.system(system_prompt),
            *extra_messages,
            final_instruction_msg,
        ]
        # messages: list[ChatMessage] = agent._loop.on_before_select_tool(
        #     messages=messages,
        # )

        # tools = get_openai_command_specs(
        #         agent._tool_registry.list_available_tools(self)
        #     ) ===== agent._tool_registry.dump_tools()
        self._function = agent._tool_registry.dump_tools()
        prompt = ChatPrompt(
            messages=messages,
            functions=self._function,
            function_call="auto",
            default_function_call="ask_user",
        )

        return prompt

    #
    # response_format_instruction
    #
    def response_format_instruction(
        self, agent: "PlannerAgent", model_name: str, **kargs
    ) -> str:
        return super().response_format_instruction(agent=agent, model_name=model_name)

    #
    # _generate_intro_prompt
    #
    def _generate_intro_prompt(self, agent: "PlannerAgent", **kargs) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return super()._generate_intro_prompt(agent, **kargs)

    #
    # _generate_os_info
    #
    def _generate_os_info(self, **kwargs) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """

        return super()._generate_os_info(**kwargs)

    #
    #     def _generate_budget_constraint
    #
    def _generate_budget_constraint(self, api_budget: float, **kargs) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        return super()._generate_budget_constraint(api_budget, **kargs)

    #
    # _generate_goals_info
    #
    def _generate_goals_info(self, goals: list[str], **kargs) -> list[str]:
        """Generates the goals information part of the prompt.

        Returns:
            str: The goals information part of the prompt.
        """
        return super()._generate_goals_info(goals, **kargs)

    #
    # _generate_tools_list
    #
    def _generate_tools_list(
        self, tools: list[CompletionModelFunction], **kargs
    ) -> str:
        """Lists the tools available to the agent.

        Params:
            agent: The agent for which the tools are being listed.

        Returns:
            str: A string containing a numbered list of tools.
        """
        return super()._generate_tools_list(tools, **kargs)

    ###
    ### parse_response_content
    ###
    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            parsed_response = json_loads(response_content["function_call"]["arguments"])
        except Exception:
            self._agent._logger.warning(parsed_response)

        ###
        ### OLD
        ###
        # parsed_response["name"] = response_content["function_call"]["name"]
        # return parsed_response

        ###
        ### NEW
        ###
        command_name = response_content["function_call"]["name"]
        command_args = parsed_response
        assistant_reply_dict = response_content["content"]

        return command_name, command_args, assistant_reply_dict

    # FIXME Move to new format
    # def parse_response_content(
    #     self,
    #     response_content: AssistantChatMessageDict,
    # ) -> dict:
    #     """Parse the actual text response from the objective model.

    #     Args:
    #         response_content: The raw response content from the objective model.

    #     Returns:
    #         The parsed response.

    #     """
    #     if "content" not in response_content:
    #         raise InvalidAgentResponseError("Assistant response has no text content")

    #     assistant_reply_dict = extract_dict_from_response(response_content["content"])

    #     try:
    #         parsed_response = json_loads(response_content["function_call"]["arguments"])
    #     except Exception:
    #         self._agent._logger.warning(parsed_response)

    #     parsed_response["name"] = response_content["function_call"]["name"]

    #     # Get command name and arguments
    #     command_name, arguments = self.extract_command(
    #         assistant_reply_dict, response, self._config.use_functions_api
    #     )
    #     return command_name, arguments, assistant_reply_dict

    #     return parsed_response

    def save(self):
        pass

    #############
    # Utilities #
    #############
    def extract_command(
        assistant_reply_json: dict,
        assistant_reply: AssistantChatMessageDict,
        use_openai_functions_api: bool,
    ) -> tuple[str, dict[str, str]]:
        super().extract_command(
            assistant_reply_json=assistant_reply_json,
            assistant_reply=assistant_reply,
            use_openai_functions_api=use_openai_functions_api,
        )

    # # NOTE : based on planning_agent.py
    # def construct_base_prompt(
    #     self, agent: "PlannerAgent", **kwargs
    # ) -> list[ChatMessage]:

    #     # Add the current plan to the prompt, if any
    #     if agent.plan:
    #         plan_section = [
    #             "## Plan",
    #             "To complete your task, you have composed the following plan:",
    #         ]
    #         plan_section += [f"{i}. {s}" for i, s in enumerate(agent.plan, 1)]

    #         # Add the actions so far to the prompt
    #         if agent.event_history:
    #             plan_section += [
    #                 "\n### Progress",
    #                 "So far, you have executed the following actions based on the plan:",
    #             ]
    #             for i, cycle in enumerate(agent.event_history, 1):
    #                 if not (cycle.action and cycle.result):
    #                     agent._logger.warn(f"Incomplete action in history: {cycle}")
    #                     continue

    #                 plan_section.append(
    #                     f"{i}. You executed the command `{cycle.action.format_call()}`, "
    #                     f"which gave the result `{cycle.result}`."
    #                 )

    #         self._prepend_messages.append(ChatMessage.system("\n".join(plan_section)))

    #     messages = super().construct_base_prompt(
    #         agent=agent,  **kwargs
    #     )

    # return messages

    def compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        start: int = len(episode_history)

        for i, c in reversed(list(enumerate(episode_history))):
            step = f"### Step {i+1}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)
            start = i

        # TODO: summarize remaining

        part = slice(0, start)

        return "\n\n".join(steps)