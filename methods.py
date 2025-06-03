import pandas as pd
from datasets import load_dataset
import re
from dotenv import load_dotenv
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import os # Added for os.getenv

from llms_api import query_model


COT_TRADITIONAL_8_SHOT_PROMPT = """
See the examples below to guide you on how your answer format should be:

Q: If John has 3 apples and buys 2 more, how many apples does he have now?
A: Let's think step by step.
John starts with 3 apples.
He buys 2 more apples.
So, the total number of apples is 3 + 2 = 5.
The answer is 5.

Q: Sarah had 10 candies. She gave 4 to her friend. How many candies does she have left?
A: Let's think step by step.
Sarah starts with 10 candies.
She gives 4 candies to her friend.
So, the number of candies left is 10 - 4 = 6.
The answer is 6.

Q: There are 5 baskets, each with 3 oranges. How many oranges are there in total?
A: Let's think step by step.
Each basket has 3 oranges.
There are 5 baskets.
So, the total number of oranges is 5 * 3 = 15.
The answer is 15.

Q: A baker has 24 cupcakes and packs them equally into 6 boxes. How many cupcakes are in each box?
A: Let's think step by step.
The baker has 24 cupcakes.
He packs them into 6 boxes equally.
So, each box contains 24 / 6 = 4 cupcakes.
The answer is 4.

Q: Emily read 7 pages of a book on Monday and 9 pages on Tuesday. How many pages did she read in total?
A: Let's think step by step.
Emily read 7 pages on Monday.
She read 9 pages on Tuesday.
So, the total number of pages read is 7 + 9 = 16.
The answer is 16.

Q: Tom had 15 marbles. He lost 7 of them. How many marbles does he have now?
A: Let's think step by step.
Tom starts with 15 marbles.
He loses 7 marbles.
So, the number of marbles left is 15 - 7 = 8.
The answer is 8.

Q: There are 4 packs of pencils, each containing 6 pencils. How many pencils are there in total?
A: Let's think step by step.
Each pack contains 6 pencils.
There are 4 packs.
So, the total number of pencils is 4 * 6 = 24.
The answer is 24.

Q: A gardener has 36 flowers and plants them equally in 9 rows. How many flowers are in each row?
A: Let's think step by step.
The gardener has 36 flowers.
He plants them equally in 9 rows.
So, each row contains 36 / 9 = 4 flowers.
The answer is 4.
"""

#TODO: REVISAR PROMPT OFICIAL MATHARENA PARA ESSE BENCH
COT_MATH_PROMPT = """"""

AIME_SHOT_EXAMPLES = r"""Please reason step by step, and put your final answer within \boxed{{}}.The answer is an integer between 0 and 999 inclusive."""

META_PROMPT_TEMPLATE = """You are an expert in adapting instructions for language models. Your task is to create a personalized Self-Reflection prompt for a model that is trying to solve a mathematical problem. You will receive the original question and should adapt the prompt based on it.
Your task is to modify the Self-Reflection template so that it is as specific and helpful as possible for the problem. Focus on aspects such as:
*   **Type of problem:** The Self-Reflection prompt should guide the model to solve the specific type of problem presented in the question.
*   **Common mistakes:** The Self-Reflection prompt should guide the model to identify the common mistakes that are made when solving this type of problem.
*   **Complexity of the problem:** The Self-Reflection prompt should guide the model to try to understand the complexity of the problem, if more steps arte needed to solve it.
Here is the original Self-Reflection template that you should adapt:
--- Beginning of the template ---
You are an expert in <PROBLEM_AREA>.
You have incorrectly answered the following question.
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future.
First, explain why you answered the question incorrectly.
Second, list the keywords that describe the type of your errors from most general to most specific.
Third, solve the problem again, step-by-step, based on your knowledge of the correct answer.
Fourth, create a list of detailed instructions to help you correctly solve this problem in the future.
Finally, create a list of general advice to help you solve similar types of problems in the future.
Be concise in your response; however, capture all of the essential information.
For guidance, I will provide you with a single generic example problem and reflection (below).
[Example Input]
Question: <an example question similar on complexity to the question received>
Wrong answer: <the wrong reasoning and answer to the example question, with a specific mistake made along the way that resulted in the wrong answer>
---
[Example Output]
Explanation:
I miscalculated the <explanation of the mistake>
Error Keywords:
- <keywords of the mistake>
Instructions:
<list of instructions to solve the problem>
Advice:
<list of general advice to solve similar types of problems>
Solution:
<the correct reasoning and answer to the example question>
--- End of the template ---
Now, adapt the above template for the following question:
Question: {question}
Generate the adapted Self-Reflection prompt (remember, you need to create a similar example question on complexity to the question received (NOT THE SAME ONE), a wrong answer to it and the correct answer):
"""



def generate_cot_prompt(question, benchmark_name):
    if benchmark_name == "MATH":
        base_prompt = COT_MATH_PROMPT
        #TODO: REVISAR PROMPT OFICIAL MATHARENA PARA ESSE BENCH
        return f"""{base_prompt} {question}"""
    elif benchmark_name == "AIME":
        base_prompt = AIME_SHOT_EXAMPLES
        return f"""{base_prompt} {question}"""
    # Default to EIGHT_SHOT_EXAMPLES for GSM types ("gsm-symbolic", "gsm8-std", "main", "p1", "p2") or any other
    else: 
        base_prompt = COT_TRADITIONAL_8_SHOT_PROMPT

    return f"""{base_prompt}
Now, look at this question:
Q: {question}
A: Let's think step by step..."""

def generate_auto_reflection_traditional_prompt(question, answer):
    return f"""You are an expert in math.
You have incorrectly answered the following question.
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future.
First, explain why you answered the question incorrectly.
Second, list the keywords that describe the type of your errors from most general to most specific.
Third, solve the problem again, step-by-step, based on your knowledge of the correct answer.
Fourth, create a list of detailed instructions to help you correctly solve this problem in the future.
Finally, create a list of general advice to help you solve similar types of problems in the future.
Be concise in your response; however, capture all of the essential information.
For guidance, I will provide you with a single generic example problem and reflection (below).
[Example Input]
Question: What is the product of the number of letters contained in the name of the city
where Iowa State University is located multiplied by the number of letters
contained in the name of the state?
Answer:
Iowa State University is located in the city of Ames
ISU is located in the state of Iowa.
The answer is 32
---
[Example Output]
Explanation:
I miscalculated the product of the number of letters in the city and state names.
The gap in my knowledge was not in geography but in basic arithmetic.
I knew the correct city and state but made a calculation error.
Error Keywords:
- Calculation error
- Arithmetic error
- Multiplication error
Instructions:
1. Identify the city where the university is located.
2. Identify the state where the university is located.
3. Count the number of letters in the name of the city.
4. Count the number of letters in the name of the state.
5. Multiply the number of letters in the city by the number of letters in the state.
6. Work step-by-step through your mathematical calculations.
7. Double-check your calculations to ensure accuracy.
8. Choose the answer that matches your calculated result.
Advice:
- Always read the question carefully and understand the problem.
- Always decompose complex problems into multiple simple steps.
- Always think through each subproblem step-by-step.
- Never skip any steps; be explicit in each step of your reasoning.
- Always double-check your calculations and final answer.
- Remember that the product of two numbers is the result of multiplying them together,
not adding them.
Solution:
Iowa State University is located in the city of Ames
Iowa State University is located in the state of Iowa.
The city name "Ames" contains 4 letters.
The state name "Iowa" contains 4 letters.
The product of 4*4 is 16.
The answer is 16
Now, look at this question:
Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections.
"""

def generate_auto_reflection_auto_adapt_prompt(question, previous_incorrect_answers, auto_prompt_model_name, auto_prompt_api_key, auto_prompt_api_provider, auto_prompt_supports_sampling_params, auto_prompt_thinking_effort_support=False, auto_prompt_reasoning_effort="medium"):
    """
    Generates an adaptive self-reflection prompt.
    The instructions for reflection are generated by a specified auto_prompt_model.
    Only previous *extracted incorrect answers* are included in the prompt.
    """
    
    current_meta_prompt = META_PROMPT_TEMPLATE.format(question=question)
    
    # Log which model is being used for meta-prompting
    print(f"[Info] Generating adaptive reflection instructions using meta-prompting model: {auto_prompt_model_name} (Provider: {auto_prompt_api_provider})")

    adapted_reflection_prompt_instructions = query_model(
        api_key=auto_prompt_api_key,
        prompt=current_meta_prompt,
        model=auto_prompt_model_name,
        supports_sampling_params=auto_prompt_supports_sampling_params,
        api_provider=auto_prompt_api_provider,
        thinking_effort_support=auto_prompt_thinking_effort_support,
        reasoning_effort=auto_prompt_reasoning_effort
    )

    if adapted_reflection_prompt_instructions is None:
        print(f"[Warning] Failed to generate adaptive reflection instructions using {auto_prompt_model_name}. Returning None.")
        return None

    # Construct the final prompt for the main model to perform reflection
    # This includes the dynamically generated instructions and the history of *extracted incorrect answers*.
    final_reflection_prompt_for_main_model = adapted_reflection_prompt_instructions
    final_reflection_prompt_for_main_model += f"""\n\nNow, look at this question:
Question: {question}
Your initial Chain-of-Thought answer (extracted result) was: {previous_incorrect_answers[0] if previous_incorrect_answers else 'None'}"""
    
    if len(previous_incorrect_answers) > 1:
        final_reflection_prompt_for_main_model += "\n\nYour previous reflection answers (extracted results) were:"
        for i, extracted_ans in enumerate(previous_incorrect_answers[1:], 1):
            final_reflection_prompt_for_main_model += f"\nReflection Attempt {i} (extracted result): {extracted_ans}"
            
    final_reflection_prompt_for_main_model += """

You previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections. Your new answer MUST be different from your previous answers because they were all incorrect."""
    
    # The detailed logging of this final_reflection_prompt_for_main_model that was here previously has been removed as per user request.
    # The caller (run_test.py) will log this prompt (or a snippet of it) if needed.

    return final_reflection_prompt_for_main_model

def generate_reanswer_prompt(question, answer, reflection):
    return f"""{COT_TRADITIONAL_8_SHOT_PROMPT}

Now, look at this question:
Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly.
Then you reflected on the problem, your solution, and the correct answer::
Reflection: {reflection}

Provide your corrected reasoning and answer in the examples format.
"""

# Function to extract the numerical answer from the GSM format
def extract_answer_gsm_format(response):
    try:
        if not response:
            return None
        response = response.replace(",", "")
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if not numbers:
            return None
        return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_answer_gsm_format: {e}")
        return None


def extract_answer_math(response_text):
    """
    Extracts the content from the last \\boxed{...} block in the response text,
    handles common natural language phrases, and applies basic LaTeX normalization.
    """
    try:
        if response_text is None:
            return None
        
        boxed_matches = re.findall(r"\\boxed{(.*)}", response_text) 
        if not boxed_matches:
             boxed_matches = re.findall(r"\\boxed{(.*?)}", response_text)

        extracted_text = ""
        if boxed_matches:
            extracted_text = boxed_matches[-1]
        else:
            lines = response_text.strip().split('\n')
            for line in reversed(lines):
                clean_line = line.strip()
                temp_line = clean_line.lower()
                common_prefixes = [
                    "the final answer is", "the answer is", "final answer:", "answer:", "solution:",
                    "so, the answer is", "thus, the answer is", "therefore, the answer is"
                ]
                for prefix in common_prefixes:
                    if temp_line.startswith(prefix):
                        temp_line = temp_line[len(prefix):].lstrip(":\s") 
                        clean_line = clean_line[len(prefix):].lstrip(":\s")
                        break
                
                if clean_line and not temp_line.startswith("let's think step by step"):
                    if (re.search(r'[\d./*\-+()\[\]ooPIsqrt]', clean_line, re.IGNORECASE) or
                        re.search(r'\\[a-zA-Z]+', clean_line) or
                        re.search(r'(?:\(|\[)-?oo|infinity', clean_line, re.IGNORECASE)): 
                        extracted_text = clean_line
                        break 
            if not extracted_text: 
                if len(response_text.split()) < 15 and (re.search(r'[\d./*\-+()\[\]ooPIsqrt]', response_text, re.IGNORECASE) or re.search(r'\\[a-zA-Z]+', response_text)):
                    extracted_text = response_text 
                else:
                    return None

        extracted_text = extracted_text.strip()
        
        phrases_to_strip_at_start = [
            "The final answer is", "The answer is", "My answer is", "So the answer is", "It is", "Final Answer:", "Answer:", "Solution:"
        ]
        for phrase in phrases_to_strip_at_start:
            if extracted_text.lower().startswith(phrase.lower()):
                extracted_text = extracted_text[len(phrase):].lstrip(":\s")

        if extracted_text.endswith("."):
            if not re.search(r"\d\.$", extracted_text): 
                 extracted_text = extracted_text[:-1]
        
        extracted_text = extracted_text.strip(".,:;!") 

        extracted_text = re.sub(r"\\text{\s*(.*?)\s*}", r"\1", extracted_text)
        extracted_text = re.sub(r"\\mathrm{\s*(.*?)\s*}", r"\1", extracted_text)

        extracted_text = re.sub(r"\\left\(", r"(", extracted_text)
        extracted_text = re.sub(r"\\right\)", r")", extracted_text)
        extracted_text = re.sub(r"\\\(", r"(", extracted_text)
        extracted_text = re.sub(r"\\\)", r")", extracted_text)
        extracted_text = re.sub(r"\\left\[", r"[", extracted_text)
        extracted_text = re.sub(r"\\right\]", r"]", extracted_text)
        extracted_text = re.sub(r"\\\[", r"[", extracted_text)
        extracted_text = re.sub(r"\\\]", r"]", extracted_text)
        extracted_text = re.sub(r"\\left\{", r"{", extracted_text)
        extracted_text = re.sub(r"\\right\}", r"}", extracted_text)

        spacing_cmds = [r"\\,", r"\\!", r"\\s", r"\\quad", r"\\qquad", r"~_*"] # Corrected \s to r"\\s"
        for cmd in spacing_cmds:
            extracted_text = extracted_text.replace(cmd, "")
        
        extracted_text = extracted_text.strip() 

        extracted_text = extracted_text.replace(r"\pi", "pi")
        extracted_text = extracted_text.replace(r"\infty", "oo") 

        extracted_text = re.sub(r"\\frac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}", r"(\1)/(\2)", extracted_text)
        extracted_text = re.sub(r"\\sqrt\s*{\s*(.*?)\s*}", r"sqrt(\1)", extracted_text)
        extracted_text = extracted_text.replace(r"\cdot", "*")
        extracted_text = extracted_text.replace(r"\times", "*")
        extracted_text = extracted_text.replace(r"^{\circ}", "") 
        extracted_text = extracted_text.replace(r"\pm", "+-") 

        if extracted_text.startswith("{") and extracted_text.endswith("}"):
            if not (re.search(r",", extracted_text[1:-1]) or re.search(r"\|", extracted_text[1:-1])):
                extracted_text = extracted_text[1:-1]
        
        if extracted_text.startswith("$") and extracted_text.endswith("$"):
            extracted_text = extracted_text[1:-1].strip()
        if extracted_text.startswith("$$") and extracted_text.endswith("$$"):
            extracted_text = extracted_text[2:-2].strip()

        return extracted_text.strip()

    except Exception as e:
        print(f"Error in extract_answer_math: {e}")
        return None


def extract_answer_aime(response_text):
    """
    Extracts the last numerical value from the response text.
    AIME answers are typically integers.
    """
    try:
        if response_text is None:
            return None
        # Replace commas to handle numbers like 1,000
        response_text = response_text.replace(",", "")
        # Find all numerical values (integers or simple decimals)
        numbers = re.findall(r"-?\d+\.?\d*", response_text)
        if numbers:
            # Get the last numerical value found
            last_num_str = numbers[-1]
            try:
                # Attempt to convert to float
                float_val = float(last_num_str)
                # If it's an integer (e.g., 16.0), convert to int string "16"
                if float_val.is_integer():
                    return str(int(float_val))
                else:
                    # Otherwise, return the original float string (e.g., "16.5")
                    return last_num_str
            except ValueError:
                # If conversion to float fails, return the original extracted string
                return last_num_str
        else:
            return None
    except Exception as e:
        print(f"Error in extract_answer_aime: {e}")
        return None