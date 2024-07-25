import openai
from langchain.llms import LLaMA
from langchain import PromptTemplate, LLMChain, Agent
from langchain.schema import BaseOutputParser
from typing import List, Tuple, Any

openai.api_key = ''

class JSONOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        import json
        return json.loads(output)

code_and_doc_prompt = PromptTemplate(
    template="Generate Python code and documentation for the following task:\n\n{task}"
)

subtask_prompt = PromptTemplate(
    template="Divide the following task into smaller subtasks:\n\n{task}"
)

combine_code_prompt = PromptTemplate(
    template="Given the following list of functions with their documentation, generate the complete solution for the task:\n\nTask: {task}\n\nFunctions:\n{functions}"
)

review_code_prompt = PromptTemplate(
    template="Review the following code and documentation for any issues or improvements:\n\nCode:\n{code}\n\nDocumentation:\n{documentation}"
)

generate_tests_prompt = PromptTemplate(
    template="Generate test cases for the following Python function:\n\nFunction:\n{code}"
)

test_code_prompt = PromptTemplate(
    template="Given the following code and test cases, run the tests and provide the results:\n\nCode:\n{code}\n\nTest Cases:\n{test_cases}"
)

llm = LLaMA(api_key=openai.api_key, model_name='llama2')
code_and_doc_chain = LLMChain(llm=llm, prompt=code_and_doc_prompt, output_parser=JSONOutputParser())
subtask_chain = LLMChain(llm=llm, prompt=subtask_prompt, output_parser=JSONOutputParser())
combine_code_chain = LLMChain(llm=llm, prompt=combine_code_prompt, output_parser=JSONOutputParser())
review_code_chain = LLMChain(llm=llm, prompt=review_code_prompt, output_parser=JSONOutputParser())
generate_tests_chain = LLMChain(llm=llm, prompt=generate_tests_prompt, output_parser=JSONOutputParser())
test_code_chain = LLMChain(llm=llm, prompt=test_code_prompt, output_parser=JSONOutputParser())

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

class ProjectAgent(Agent):
    def __init__(self, code_and_doc_chain: LLMChain, subtask_chain: LLMChain, combine_code_chain: LLMChain, review_code_chain: LLMChain, generate_tests_chain: LLMChain, test_code_chain: LLMChain):
        self.code_and_doc_chain: LLMChain = code_and_doc_chain
        self.subtask_chain: LLMChain = subtask_chain
        self.combine_code_chain: LLMChain = combine_code_chain
        self.review_code_chain: LLMChain = review_code_chain
        self.generate_tests_chain: LLMChain = generate_tests_chain
        self.test_code_chain: LLMChain = test_code_chain

    def run(self, task_description: str) -> Tuple[str, str]:
        root_node = Node(task_description)

        self.divide_task(root_node)

        self.recursive_generate_and_assemble(root_node)

        project_code, project_doc = self.assemble_solution(root_node)

        return project_code, project_doc

    def divide_task(self, node: Node) -> None:
        node.divide_task()
        for child in node.children:
            self.divide_task(child)

    def recursive_generate_and_assemble(self, node: Node) -> None:
        node.generate_code_and_doc()
        
        for child in node.children:
            self.recursive_generate_and_assemble(child)

        node.review_and_test_code()

    def assemble_solution(self, node: Node) -> Tuple[str, str]:
        if not node.children:
            return node.code, node.documentation

        functions_doc: str = ""
        for child in node.children:
            child_code, child_doc = self.assemble_solution(child)
            functions_doc += f"Function: {child_code}\nDocumentation: {child_doc}\n\n"

        response: str = self.combine_code_chain.run(task=node.description, functions=functions_doc)
        node.code = response.strip()

        return node.code, node.documentation

def main() -> None:
    agent = ProjectAgent(
        code_and_doc_chain=code_and_doc_chain,
        subtask_chain=subtask_chain,
        combine_code_chain=combine_code_chain,
        review_code_chain=review_code_chain,
        generate_tests_chain=generate_tests_chain,
        test_code_chain=test_code_chain
    )

    root_task: str = "Create a web application with authentication, database, and API endpoints"

    project_code, project_doc = agent.run(root_task)

    print("Generated Project Code:\n")
    print(project_code)
    print("\nGenerated Project Documentation:\n")
    print(project_doc)

if __name__ == "__main__":
    main()
