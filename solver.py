from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from problem import Problem
from generalist import Generalist
from critic import Critic
from coder import Coder
from utils import str_to_python_func

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

        """
        1. Analyze the problem at hand and break it down into pieces.
        """
        
        # Step 1: Analyze the problem with Generalist and Critic
        problem_understanding_prompt = problem_understading_generalist_template.format(
            problem_description=problem.description
        )
        problem_understanding = generalist.invoke([problem_understanding_prompt])
        problem_verbose_description = problem_understanding.content
        
        feedback_prompt = problem_understanding_critic_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description
        )
        feedback = critic.invoke([feedback_prompt])
        feedback_text = feedback.content
        
        revised_analysis_prompt = problem_understanding_feedback_generalist_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description,
            critic_analysis_feedback=feedback_text
        )
        revised_analysis = generalist.invoke([revised_analysis_prompt])
        problem.verbose_description = revised_analysis.content

        # Generate subproblems
        problem.subproblems = problem.generate_subproblems(generalist)
        
        # Process each subproblem recursively
        for subproblem in problem.subproblems:
            self.invoke(subproblem, generalist, critic, coder)

        """
        2. Generate candidate verbal solutions.
        """

        for _ in range(problem.num_of_candidate_solutions):
            candidate_solution_prompt = candidate_verbal_solution_generalist_template.format(
                problem_verbose_description=problem.verbose_description
            )
            candidate_solution = generalist.invoke([candidate_solution_prompt])
            problem.candidate_solutions.append(candidate_solution.content)
        

        """
        3. Analyze candidate verbal solutions via Critic.
        """

        candidate_reviews = []
        for solution in problem.candidate_solutions:
            candidate_review_prompt = candidate_verbal_solution_critic_template.format(
                problem_verbose_description=problem.verbose_description,
                problem_verbal_solution=solution
            )
            review = critic.invoke([candidate_review_prompt])
            candidate_reviews.append(review.content)
        
        # Select the best solution based on feedback
        best_solution_index = candidate_reviews.index(max(candidate_reviews))
        problem.chosen_solution = problem.solutions[best_solution_index]

        """
        4. Feedback those critiques to the Generalist.
        """

        feedback_prompt = candidate_verbal_solution_feedback_generalist_template.format(
            problem_verbose_description=problem.verbose_description,
            problem_verbal_solution=problem.chosen_solution,
            problem_verbal_solution_feedback=candidate_reviews[best_solution_index]
        )
        revised_solution = generalist.invoke([feedback_prompt])
        problem.chosen_solution = revised_solution.content

        """
        5. Generate subproblems for coding.
        """

        subproblems_prompt = "Generate a list of pure functions with verbose, descriptive names needed to implement the solution:\nSolution: " + problem.chosen_solution
        subproblems = generalist.invoke([subproblems_prompt])
        subproblem_functions = subproblems.content
        
        # Parse each function and generate new Problems
        for func_code in subproblem_functions.split('\n'):
            function = str_to_python_func(func_code)
            if isinstance(function, Exception):
                continue  # Handle exceptions if needed
            subproblem_description = f"Implement the function: {function.__name__}"
            subproblem = Problem(description=subproblem_description)
            problem.subproblems.append(subproblem)

        """
        6. Parse each subproblem, initialize new `Problem` objects, 
           and add them to the current `Problem` object.
        """

        # This is done in step 5 as each subproblem is initialized and added to `problem.subproblems`

        """
        7. Repeat steps 1:6 for each subproblem until the leaves are reached.
        """
        
        return problem

        
