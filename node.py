from typing import List, Tuple, Any
from langchain.llms import LLaMA
from langchain import PromptTemplate, LLMChain, Agent
from langchain.schema import BaseOutputParser
from langchain import PromptTemplate, LLMChain, Agent

class Node:
    def __init__(self, description: str):
        self.description: str = description
        self.children: List['Node'] = []
        self.code: str = ""
        self.documentation: str = ""
        self.test_cases: str = ""

    def add_child(self, child: 'Node') -> None:
        self.children.append(child)

    def generate_code_and_doc(self) -> None:
        response: str = code_and_doc_chain.run(task=self.description)
        self.code, self.documentation = self.parse_response(response)

    def parse_response(self, response: str) -> Tuple[str, str]:
        parts: List[str] = response.split("Documentation:")
        code: str = parts[0].strip()
        documentation: str = parts[1].strip() if len(parts) > 1 else ""
        return code, documentation

    def divide_task(self) -> None:
        response: str = subtask_chain.run(task=self.description)
        subtasks: List[str] = self.parse_subtasks(response)
        for subtask in subtasks:
            child_node = Node(subtask)
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
        while True:
            review_response: str = review_code_chain.run(code=self.code, documentation=self.documentation)
            self.code, self.documentation = self.apply_review_feedback(review_response)
            
            test_response: str = generate_tests_chain.run(code=self.code)
            self.test_cases = test_response.strip()

            test_result: str = test_code_chain.run(code=self.code, test_cases=self.test_cases)
            if "All tests passed" in test_result:
                break
            else:
                print("Test failed, regenerating code based on feedback...")
                self.code, self.documentation = self.apply_review_feedback(test_result)

    def apply_review_feedback(self, response: str) -> Tuple[str, str]:
        parts: List[str] = response.split("Documentation:")
        code: str = parts[0].strip()
        documentation: str = parts[1].strip() if len(parts) > 1 else ""
        return code, documentation
