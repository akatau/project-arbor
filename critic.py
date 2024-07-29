from langchain import PromptTemplate, LLMChain, Agent
from typing import List, Tuple
import openai


class Critic:
    def __init__(self, review_code_chain: LLMChain, generate_tests_chain: LLMChain, test_code_chain: LLMChain):
        self.review_code_chain: LLMChain = review_code_chain
        self.generate_tests_chain: LLMChain = generate_tests_chain
        self.test_code_chain: LLMChain = test_code_chain

    def review_and_test_code(self, code: str, documentation: str) -> Tuple[str, str, str]:
        while True:
            review_response: str = self.review_code_chain.run(code=code, documentation=documentation)
            code, documentation = self.apply_review_feedback(review_response)
            
            test_response: str = self.generate_tests_chain.run(code=code)
            test_cases = test_response.strip()

            test_result: str = self.test_code_chain.run(code=code, test_cases=test_cases)
            if "All tests passed" in test_result:
                break
            else:
                print("Test failed, regenerating code based on feedback...")
                code, documentation = self.apply_review_feedback(test_result)
        
        return code, documentation, test_cases

    def apply_review_feedback(self, response: str) -> Tuple[str, str]:
        parts: List[str] = response.split("Documentation:")
        code: str = parts[0].strip()
        documentation: str = parts[1].strip() if len(parts) > 1 else ""
        return code, documentation

if __name__ == "__main__":
    from main import review_code_chain, generate_tests_chain, test_code_chain

    critic = Critic(
        review_code_chain=review_code_chain,
        generate_tests_chain=generate_tests_chain,
        test_code_chain=test_code_chain
    )

    sample_code = """
    def add(a, b):
        return a + b
    """
    sample_doc = """
    This function adds two numbers and returns the result.
    """

    reviewed_code, reviewed_doc, test_cases = critic.review_and_test_code(sample_code, sample_doc)

    print("Reviewed Code:\n", reviewed_code)
    print("\nReviewed Documentation:\n", reviewed_doc)
    print("\nGenerated Test Cases:\n", test_cases)
