import enum
from logging import Logger
from typing import Optional


from autogpts.AFAAS.app.lib import Task
from autogpts.autogpt.autogpt.core.prompting.base import (
    BasePromptStrategy, LanguageModelClassification, PromptStrategiesConfiguration)
from autogpts.autogpt.autogpt.core.prompting.utils.utils import (
    json_loads, to_numbered_list)
from autogpts.autogpt.autogpt.core.resource.model_providers import (
    AssistantChatMessageDict, ChatMessage, ChatPrompt, CompletionModelFunction)
from autogpts.autogpt.autogpt.core.utils.json_schema import JSONSchema


class InitialPlanFunctionNames(str, enum.Enum):
    INITIAL_PLAN: str = "make_initial_plan"


class InitialPlanStrategyConfiguration(PromptStrategiesConfiguration):
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.SMART_MODEL_8K
    )
    default_function_call: InitialPlanFunctionNames = (
        InitialPlanFunctionNames.INITIAL_PLAN
    )


class InitialPlanStrategy(BasePromptStrategy):
    default_configuration = InitialPlanStrategyConfiguration()
    STRATEGY_NAME = "make_initial_plan"

    ###
    ### PROMPTS
    ###

    FIRST_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. You're responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. Your job is to "
        "break down those goals into a set of tasks that the agent can accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions. Each task you create should have clearly defined "
        "`ready_criteria` that the agent can check to see if the task is ready to be started. Each task should "
        "also have clearly defined `acceptance_criteria` that the agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish the goals.\n\n"
        "System Info:\n{system_info}"
    )

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}\n" "Your goals are:\n" "{agent_goals}"
    )

    ###
    ### FUNCTIONS
    ###

    DEFAULT_CREATE_PLAN_FUNCTION = CompletionModelFunction(
        name=InitialPlanFunctionNames.INITIAL_PLAN.value,
        description="Creates a set of tasks that forms the initial plan for an autonomous agent.",
        parameters={
            "task_list": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "task_goal": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="The main goal or purpose of the task.",
                        ),
                        "long_description": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="A detailed description of the task.",
                        ),
                        "task_context": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Additional context or information about the task.",
                        ),
                        "acceptance_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A list of measurable and testable criteria that must be met for the task to be considered complete.",
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    def __init__(
        self,
        logger: Logger,
        model_classification: LanguageModelClassification,
        default_function_call: InitialPlanFunctionNames,
        temperature: float,  # if coding 0.05
        top_p: Optional[float],
        max_tokens: Optional[int],
        frequency_penalty: Optional[float],  # Avoid repeting oneselfif coding 0.3
        presence_penalty: Optional[float],  # Avoid certain subjects
    ):
        self._logger = logger
        self._model_classification = model_classification

        self._system_prompt_template = self.FIRST_SYSTEM_PROMPT_TEMPLATE
        self._system_info = self.DEFAULT_SYSTEM_INFO
        self._user_prompt_template = self.DEFAULT_USER_PROMPT_TEMPLATE
        self._strategy_functions = [self.DEFAULT_CREATE_PLAN_FUNCTION]

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        agent_goal_sentence: str,
        tools: list[str],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    ) -> ChatPrompt:
        template_kwargs = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }
        template_kwargs["agent_goals"] = (
            to_numbered_list(agent_goals, **template_kwargs),
        )
        template_kwargs["agent_goal_sentence"] = (agent_goal_sentence,)
        template_kwargs["tools"] = to_numbered_list(tools, **template_kwargs)
        template_kwargs["system_info"] = to_numbered_list(
            self._system_info, **template_kwargs
        )

        system_prompt = ChatMessage.system(
            content=self._system_prompt_template.format(**template_kwargs),
        )
        user_prompt = ChatMessage.user(
            content=self._user_prompt_template.format(**template_kwargs),
        )
        strategy_functions = self._strategy_functions

        response_format_instr = ChatMessage.system(
            self.response_format_instruction(
                agent=self._agent,
                model_name=kwargs["model_name"],
            )
        )

        return ChatPrompt(
            messages=[system_prompt, user_prompt, response_format_instr],
            functions=strategy_functions,
            function_call=InitialPlanFunctionNames.INITIAL_PLAN.value,
            default_function_call=InitialPlanFunctionNames.INITIAL_PLAN.value,
            # TODO:
            tokens_used=0,
        )

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
        parsed_response = json_loads(response_content["function_call"]["arguments"])
        parsed_response["task_list"] = [
            Task.parse_obj(task) for task in parsed_response["task_list"]
        ]
        return parsed_response