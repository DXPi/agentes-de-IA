import logging
from typing import Optional
from forge.sdk.model import TaskRequestBody
from autogpt.config.config import ConfigBuilder
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.agent_manager.agent_manager import AgentManager
from autogpt.agents.agent_member import AgentMember, AgentMemberSettings
from autogpt.agent_factory.profile_generator import AgentProfileGenerator
from autogpt.core.resource.model_providers.schema import ChatModelProvider

logger = logging.getLogger(__name__)

class AgentGroup:
    
    leader: AgentMember
    members: dict[str, AgentMember]

    def assign_group_to_members(self):
        self.leader.recursive_assign_group(self)

    def reload_members(self):
        members = self.leader.get_list_of_all_your_team_members()
        members_dict = {}
        for agent_member in members:
            members_dict[agent_member.id] = agent_member
        self.members = members_dict
        
    def __init__(
        self,
        leader: AgentMember
    ):
        self.leader = leader
        self.assign_group_to_members()
        self.reload_members()

    async def create_task(self, task: TaskRequestBody):
        await self.leader.create_task(task)

async def create_agent_member(
    role: str,
    initial_prompt:str,
    llm_provider: ChatModelProvider,
    boss: Optional['AgentMember'] = None,
    recruiter: Optional['AgentMember'] = None,
    create_agent: bool = False,
) -> Optional[AgentMember]:
    config = ConfigBuilder.build_config_from_env()
    config.logging.plain_console_output = True

    config.continuous_mode = False
    config.continuous_limit = 20
    config.noninteractive_mode = True
    config.memory_backend = "no_memory"
    settings = await generate_agent_settings_for_task(
            task=initial_prompt,
            app_config=config,
            llm_provider=llm_provider
        )

    agent_member = AgentMember(
        role=role,
        initial_prompt=initial_prompt,
        settings=settings,
        boss=boss,
        recruiter=recruiter,
        create_agent=create_agent,
        llm_provider=llm_provider,
    )

    agent_manager = AgentManager(config.app_data_dir)
    agent_member.attach_fs(agent_manager.get_agent_dir(agent_member.id))

    if boss:
        boss.members.append(agent_member)

    return agent_member

async def generate_agent_settings_for_task(
    task: str,
    llm_provider: ChatModelProvider,
    app_config
) -> AgentMemberSettings:
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    prompt = agent_profile_generator.build_prompt(task)
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)


    return AgentMemberSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=ai_directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
            plugins=app_config.plugins,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )
