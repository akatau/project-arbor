from typing import Optional, List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from problem import Problem
from generalist import Generalist
from critic import Critic
from coder import Coder
from utils import str_to_python_func
import json

class Solver(RunnableSerializable):
    """
    Solver is a `Runnable` that implements the entire pipeline of the prompting technique.

    It should be initialized with the parameters of the pipelines 
    (e.g., number of candidate solutions, number of reviews, etc.).
    """
    
    def generate_subproblems(self, problem: Problem, generalist_model: Generalist) -> List[str]:
        """Prompts `generalist_model` to figure out practical subproblems."""
        # Initialize subproblems list
        problem.subproblems = []
        
        solution_suproblems_analysis_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Given this problem and proposed solution, can the solution be practically divided into\
            standalone functions? If so, write a list of such subproblems and explain them. Note: Every block of \
            code (for-loops, while-loops, exception handling, etc) must be wrapped around \
            a function with with a verbose descriptive names. If it can't be divided, simply answer with No. 
            """
        )
        
        solution_suproblems_list_geneartion_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Problem's Tools/Subproblems description: {problem_subproblems_analysis}
            Given this problem, and solution, and, rewrite the list of tools/parts necessary to solve the problem \
            into a list of problems with clear problem description. Separate between suproblems with (---)
            i.e.
                subproblem1 --- subproblem --- ....
            """
        )
        
        # Invoke the generalist model to get potential subproblems
        ret = generalist_model.invoke(
            solution_suproblems_analysis_generalist_template.format(
                problem_verbose_description=problem.verbose_description, 
                problem_verbal_solution=problem.chosen_solution,
            )
        ).content
        
        if ret == "No":
            return None
        
        ret = generalist_model.invoke(
            solution_suproblems_list_geneartion_generalist_template.format(
                problem_verbose_description=problem.verbose_description, 
                problem_verbal_solution=problem.chosen_solution,
                problem_subproblems_analysis = ret
            )
        ).content
        
        # If the model returns valid subproblems, parse and collect them
        if ret:
            problem.subproblems = [Problem(subproblem, problem.num_of_candidate_solutions, problem.type) for subproblem in ret.split("---")] # Assume the returned format is a list of subproblem
        
        return problem.subproblems
    
    def invoke(
            self, 
            problem: Problem, 
            generalist: Generalist,
            critic: Critic,
            coder: Coder,
            config: Optional[RunnableConfig] = None
            ) -> Tuple[Problem, dict]:

        problem_understading_generalist_template = PromptTemplate.from_template(
            """Analyze the following problem and break it down to pieces 
            while considering the possible edge cases without providing code:\nProblem: {problem_description}""")
        
        problem_understanding_critic_template = PromptTemplate.from_template(
            """Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Review the break down for the given problem without providing code.
            Does it cover all factors and parameters in the problem? 
            How can this analysis be improved? Explain thoroughly.
            """
        )

        problem_understanding_feedback_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_description}
            Analysis: {problem_verbose_description}
            Feedback: {critic_analysis_feedback}
            Based on the feedback and critique of the problem analysis, give a more comprehensive analysis to the problem without providing code.
            """
        )

        candidate_verbal_solution_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Available functions to used: {returned_functions}
            without providing code, how can this problem be solved? Go throught the solution step by step, and
            the solution must be verbal and different from the following solutions {previous}
            """
        )

        candidate_verbal_solution_critic_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Available functions to used: {returned_functions}
            Solution: {problem_verbal_solution}
            without providing code, does this solution solves the problem correctly while coering the edge cases? Why?
            How can it be improved? Explain your answer and break it down thoroughly.
            """
        )

        candidate_verbal_solution_feedback_generalist_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Solution: {problem_verbal_solution}
            Available functions to used: {returned_functions}
            Feedback: {problem_verbal_solution_feedback}
            without providing code, rewrite the verbal solution to the given problem based on this feedback.
            """
        )
        
        best_candidate_verbal_solution_critic_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Verbal_candidate_solutions: {candidate_solutions}
            Available functions to used: {returned_functions}
            without providing code, choose the best verbal solution among them that addresses 
            the problem and covering all its aspects, you must only return the best verbal 
            solution among them without adding anything else or explaining
            
            i.e.
                Problem: Sort an array
                Verbal_candidate_solutions: ["Bubble Sort", "Merge Sort"]

                your output must be "Merge Sort" because it runs in O(n log(n)) and the other
                in O(n ^ 2)
            """
        )
        
        code_provider_for_verbal_solution_coder_template = PromptTemplate.from_template(
            """Problem: {problem_verbose_description}
            Available functions to used: {returned_functions}
            Verbal Solution: {problem_verbal_solution}
            Provide the python implementation with its docstring of the given verbal solution 
            for the given problem            
            """
        )
        
        function_header_and_docstring_extractor_generalist = PromptTemplate.from_template(
            """Code: {code}
            Extract the function header and the docstring of how to use the function, and
            this is only what must be included in your response
            
            i.e.
                def add(a: int, b: int) -> int:
                  '''Python function to add two numbers
                    a: int
                    b: int
                    
                    ret: int, which is the sum of a and b
                  '''
                  
                  return a + b
                  
                your response must be only the following
                def(a: int, b: int) -> int, and its docstring is 
                '''Python function to add two numbers
                    a: int
                    b: int
                    
                    ret: int, which is the sum of a and b
                  '''
            """
        )
        
        """
        1. Analyze the problem at hand and break it down into pieces.
        """
        
        # Use the generalist model to generate to analyse the problem further and generate verbose description
        problem_understanding_prompt = problem_understading_generalist_template.format(
            problem_description=problem.description
        )
        problem_understanding = generalist.invoke([problem_understanding_prompt])
        problem_verbose_description = problem_understanding.content
        
        # Review the problem's description and verbose description and provide enhancements for the analysis
        feedback_prompt = problem_understanding_critic_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description
        )
        feedback = critic.invoke([feedback_prompt])
        feedback_text = feedback.content
        
        # Use the problem's description, verbose description and feedback from the critic to generate the final problem's verbose description
        revised_analysis_prompt = problem_understanding_feedback_generalist_template.format(
            problem_description=problem.description,
            problem_verbose_description=problem_verbose_description,
            critic_analysis_feedback=feedback_text
        )
        revised_analysis = generalist.invoke([revised_analysis_prompt])
        problem.verbose_description = revised_analysis.content

        # Generate subproblems
        problem.subproblems = self.generate_subproblems(problem, generalist)
        
        returned_functions_and_docs = []
        Tree = {
            "Problem": problem.description,
            "code": None,
            "subproblems": []
        }
        # Process each subproblem recursively
        if problem.subproblems != None:
            for subproblem in problem.subproblems:
                returned_problem_and_tree = self.invoke(subproblem, generalist, critic, coder)
                returned_functions_and_docs.append(returned_problem_and_tree[0].chosen_solution)
                Tree["subproblems"].append(returned_problem_and_tree[1])
        """
        2. Generate candidate verbal solutions.
        """

        for _ in range(problem.num_of_candidate_solutions):
            #Generate candididate solution for the problem that's different from the previous candidate solutions
            candidate_solution_prompt = candidate_verbal_solution_generalist_template.format(
                problem_verbose_description=problem.verbose_description,
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                previous= problem.candidate_solutions if len(problem.candidate_solutions) > 0 else "No previous solutions"
            )
            candidate_solution = generalist.invoke([candidate_solution_prompt])
            problem.candidate_solutions.append(candidate_solution.content)
        

        """
        3. Analyze candidate verbal solutions via Critic.
        """
        # Generate a review for each candidate solution to fix it and ehance it
        candidate_reviews = []
        for solution in problem.candidate_solutions:
            candidate_review_prompt = candidate_verbal_solution_critic_template.format(
                problem_verbose_description=problem.verbose_description,
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                problem_verbal_solution=solution
            )
            review = critic.invoke([candidate_review_prompt])
            candidate_reviews.append(review.content)

        """
        4. Feedback those critiques to the Generalist.
        """

        # Given the description of the problem, candidate solution and its review, this part works on improving the candidate solution
        enhanced_candidate_solutions = []
        for i in range(problem.num_of_candidate_solutions):
            feedback_prompt = candidate_verbal_solution_feedback_generalist_template.format(
                problem_verbose_description=problem.verbose_description,
                problem_verbal_solution=problem.candidate_solutions[i],
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                problem_verbal_solution_feedback=candidate_reviews[i]
            )
            revised_solution = generalist.invoke([feedback_prompt])
            enhanced_candidate_solutions.append(revised_solution.content)

        """
        5. Choose the solution that best addresses the problem without missing any of its aspects
        """
         
        problem.chosen_solution = critic.invoke([
            best_candidate_verbal_solution_critic_template.format(
                problem_verbose_description=problem.verbose_description,
                candidate_solutions=enhanced_candidate_solutions
            )
        ]).content
        
        
        """
        6. Generate the code for the chosen solution.
        """
        
        problem.chosen_solution = coder.invoke([
            code_provider_for_verbal_solution_coder_template.format(
                problem_verbose_description=problem.verbose_description,
                returned_functions=returned_functions_and_docs if len(returned_functions_and_docs) > 0 else "There are no functions that can used",
                candidate_solutions=problem.chosen_solution
            )
        ]).content
        
        """
        7. Parse the output
        """
        
        problem.chosen_solution = str_to_python_func(problem.chosen_solution)
        Tree["code"] = problem.chosen_solution
        
        """
        8. Extract the function header and the docstring
        """
        
        problem.chosen_solution = generalist.invoke([
            function_header_and_docstring_extractor_generalist.format(
                code=problem.chosen_solution
            )
        ]).content
        
       

        return problem

if __name__ == '__main__':
    client = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    solve = Solver()
    ret = solve.invoke(Problem("""Python function to add 2 numbers""", 1, "Math"), client, client, client, 1)
    Project_Graph = json.dumps(ret[1], indent=4)
    
    print(Project_Graph)
