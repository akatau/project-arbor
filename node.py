from typing import List, Tuple, Any
from langchain.llms import LLaMA
from langchain import PromptTemplate, LLMChain, Agent
from langchain.schema import BaseOutputParser
from langchain import PromptTemplate, LLMChain, Agent
from critic import Critic

class Node:
    def __init__(self, description: str, critic: Critic, code_and_doc_chain : LLMChain, subtask_chain : LLMChain):
        self.code_and_doc_chain = code_and_doc_chain
        self.subtask_chain = subtask_chain
        self.description: str = description
        self.children: List['Node'] = []
        self.code: str = ""
        self.documentation: str = ""
        self.test_cases: str = ""
        self.critic: Critic = critic

    def add_child(self, child: 'Node') -> None:
        self.children.append(child)

    def generate_code_and_doc(self) -> None:
        response: str = self.code_and_doc_chain.run(task=self.description)
        self.code, self.documentation = self.parse_response(response)

    def parse_response(self, response: str) -> Tuple[str, str]:
        parts: List[str] = response.split("Documentation:")
        code: str = parts[0].strip()
        documentation: str = parts[1].strip() if len(parts) > 1 else ""
        return code, documentation

    def divide_task(self) -> None:
        response: str = self.subtask_chain.run(task=self.description)
        subtasks: List[str] = self.parse_subtasks(response)
        for subtask in subtasks:
            child_node = Node(subtask, self.critic)
            self.add_child(child_node)
            child_node.divide_task()

    def parse_subtasks(self, response: str) -> List[str]:
        return [line.strip() for line in response.split("\n") if line.strip()]

    def assemble_project(self) -> Tuple[str, str]:
        project_code: str = ""
        project_doc: str = self.documentation + "\n\n"
        for child in self.children:
            child_code, child_doc = child.assemble_project()
            project_code += child_code + "\n"
            project_doc += child_doc + "\n"
        project_code += self.code
        return project_code, project_doc

    def review_and_test_code(self) -> None:
        self.code, self.documentation, self.test_cases = self.critic.review_and_test_code(self.code, self.documentation)
