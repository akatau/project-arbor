from typing import List, Tuple, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from generalist import Generalist
from coder import Coder



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
        self.candidate_solutions: List[str] = None
        """List of candidate solutions to the problem."""
        self.chosen_solution: str = None
        """Chosen solution from candidate generated solutions."""
        self.test_cases: List[Tuple[str, bool]] = None
        """LLM-generated test cases and `chosen_solution` scores."""
        self.problem_solution_examples: List[Tuple[str, str]] = problem_solution_examples
        """LLM-generated list of problem-solution examples to the given problem for few-shot learning."""
        self.num_of_candidate_solutions = 0

    def generate_subproblems(self, generalist_model: Generalist) -> List[str]:
        """Prompts `generalist_model` to figure out practical subproblems."""
        # Initialize subproblems list
        self.subproblems = []

        # Define the prompt to divide the problem into subproblems
        dividing_problem_into_smaller_subproblems_prompt = PromptTemplate.from_template(
            """Understand carefully the following problem: {problem}
            and its verbose description: {description}.
            Decide if it can be divided into practical subproblems.
            If so, provide them in the following format: [[[ (subproblem1), (subproblem2), .... ]]]
            """
        )
        
        # Invoke the generalist model to get potential subproblems
        ret = generalist_model.invoke(
            dividing_problem_into_smaller_subproblems_prompt.format(
                problem=self.description, 
                description=self.verbose_description or self.description,
            )
        ).content
        
        # If the model returns valid subproblems, parse and collect them
        if ret:
            new_subproblems = eval(ret)  # Assume the returned format is a list of subproblems
            for subproblem in new_subproblems:
                if subproblem not in self.subproblems:
                    self.subproblems.append(subproblem)
        
        return self.subproblems

                
    def generate_verbose_description(self, generalist_model: Generalist) -> str:
        """Generate the verbose analysis of the problem."""
        verbose_description_of_the_problem_prompt = PromptTemplate.from_template(
            """Given the following problem description: {description},
            provide a verbose description of the problem, its characteristics, 
            potential edge cases, and how it can be tested.
            """
        )
        
        result = generalist_model.invoke(
            verbose_description_of_the_problem_prompt.format(
                description=self.description
            )
        ).content
        
        self.verbose_description = result
        return self.verbose_description


    def generate_solutions(self, coder_model: Coder) -> List[str]:
        """Generate candidate solutions to the problem."""
        # Initialize solutions list
        self.solutions = []
        
        # Define the prompt to generate solutions for the problem
        candidate_solutions_prompt = PromptTemplate.from_template(
            """Given the following problem description: {description} and its verbose analysis: {verbose_description}, 
            provide a list of candidate solutions to address the problem effectively.
            Your response must be in the following format
            [(solution1), (solution2), ....]
            """
        )
        
        # Invoke the coder model to generate candidate solutions
        ret = coder_model.invoke(
            candidate_solutions_prompt.format(
                description=self.description, 
                verbose_description=self.verbose_description or self.description,
            )
        ).content
        
        # Assume the model returns a formatted list of solutions
        if ret:
            new_solutions = eval(ret)  # Assuming the format is a list of solutions
            for solution in new_solutions:
                if solution not in self.solutions:
                    self.solutions.append(solution)
        
        return self.solutions
