from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from problem import Problem
from generalist import Generalist
from critic import Critic
from coder import Coder

class Solver(RunnableSerializable):
    """
    Solver is a `Runnable` that implements the entire pipeline of the prompting technique.

    It should be initialized with the parameters of the pipelines 
    (e.g., number of candidate solutions, number of reviews, etc.).
    """
    def invoke(
            self, 
            problem: Problem, 
            generalist: Generalist,
            critic: Critic,
            coder: Coder,
            config: Optional[RunnableConfig] = None
            ) -> Problem:

        """
        1. Analyze the problem at hand and break it down to pieces :: RunnableSequence
        """

        
        problem_understading_generalist_template = PromptTemplate.from_template(
            """Analyze the following problem and break it down to pieces 
            while considering the possible edge cases:\nProblem: {problem_description}""")
        
        problem_understanding_critic_template = PromptTemplate.from_template(
            """Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Review the break down for the given problem.
            Does it cover all factors and parameters in the problem? 
            How can this analysis be improved? Explain thoroughly.
            """
        )

        problem_understanding_feedback_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Feedback: {critic_analysis_feedback}
            Based on the feedback and critique of the problem analysis, give a more comprehensive analysis to the problem.
            """
        )

        candidate_verbal_solution_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            How can this problem be solved? Go throught the solution step by step.
            """
        )

        candidate_verbal_solution_critic_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Does this solution solves the problem correctly while coering the edge cases? Why?
            How can it be improved? Explain your answer and break it down thoroughly.
            """
        )

        candidate_verbal_solution_feedback_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Feedback: {problem_verbal_solution_feedback}
            Rewrite the solution to the given problem based on this feedback.
            """
        )

        
        
        understand_problem = generalist | critic

        """

        2. Generate candidate verbal solutions :: RunnableSequence
        
        3. Analyze candidate verbal solutions via Critic :: RunnableSequence
        
        4. Feedback those critiques to the Generalist to generate other candidate 
           verbal solutions accordingly :: RunnableSequence
           NOTE: This may typically be repeated untill at least one verbal solution 
           is accepted by the Critic :: RunnableSequence
        
        5. Pass the problem with the chosen verbal solution to the Generalist to give a list of
           pure functions to generate with very verbose, descriptive names necessary to solve 
           implement the chosen verbal solution: the sub problems.
           Possibly also pass the output to Critic.
           Make sure every block (like for-loops, while-loops, etc) or chunk is wrapped around a function.

        6. Parse each subproblem, initialize new `Problem` objects with them, 
           add the new `Problem` objects to the current `Problem` object.

        7. Repeat steps 1:6 for each subproblem until the leaves are reached.
        """

        