"""Everything an agent needs: config, the Model gateway, the RAG client and the chosen System 1 decider."""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from agentic_jev_rag.config import Config
from agentic_jev_rag.gateway import ModelGateway
from agentic_jev_rag.recruiter.agent_loop import Tool
from agentic_jev_rag.recruiter.rag_client import RagClient
from agentic_jev_rag.recruiter.tools import rag_tools
from agentic_jev_rag.system1 import System1Decider, make_deciders


@dataclass
class RecruiterContext:
    config: Config
    gateway: ModelGateway
    rag: RagClient
    system_1: str = "llm"
    today: Callable[[], date] = field(default=date.today)

    @property
    def decider(self) -> System1Decider:
        return make_deciders(self.config, self.gateway)[self.system_1]

    def tools(self, *names: str) -> list[Tool]:
        """RAG tools by name; `ask` uses this context's System 1 decider."""
        tools = rag_tools(self.rag)
        ask = tools["ask"]
        tools["ask"] = Tool(ask.name, ask.description, ask.parameters, lambda **kw: self.rag.ask(system_1=self.system_1, **kw))
        return [tools[name] for name in names]
