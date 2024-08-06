from typing import List, Tuple, Any
import json

from langchain_core.language_models import BaseChatModel
from langchain.llms import LLaMA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain import PromptTemplate, LLMChain, Agent
from langchain.schema import BaseOutputParser
from langchain import PromptTemplate, LLMChain, Agent



class Problem(BaseMessage):
    """
    `Problem` is a `BaseMessage` object type that has all the information about a given problem.

    `Problem` is a subclass of `BaseMessage` instead of `LanguageModelInput` because it is both expected
    to both be passed to and *returned* by `BaseChatModel` objects.
    """

    def __init__(self,
                 description: str, 
                 problem_solution_examples: List[Tuple[str, str]]= None,
                 **kwargs: Any
                 ):
        """
        """
        super().__init__(content=description)

        self.description: str = self.content
        "Description of the problem."
        self.verbose_description: str = None
        """LLM-generated verbose description of the problem, its characteristics, and edge cases
        —also possibly how it can be tested."""
        self.subproblems: List['Problem'] = None
        """List of all subproblems of the problem."""
        self.solutions: List[str] = None
        """List of candidate solutions to the problem."""
        self.chosen_solution: str = None
        """Chosen solution from candidate generated solutions."""
        self.test_cases: List[Tuple[str, bool]] = None
        """LLM-generated test cases and `chosen_solution` scores."""
        self.problem_solution_examples: List[Tuple[str, str]] = problem_solution_examples
        """LLM-generated list of problem-solution examples to the given problem for few-shot learning."""

    def generate_subproblems(self, generalist_model:BaseChatModel) -> List[str]:
        """Prompts `generalist_model` to figure out practical subproblems. To be implemented"""
        pass

    def generate_verbose_description(self, generalist_model: BaseChatModel) -> str:
        """Generate the verbose analysis of the problem."""
        pass

    def generate_solutions(self, coder_model:BaseChatModel) -> List[str]:
        """Generate candidate solutions to the problem."""
        pass
