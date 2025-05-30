import pandas as pd
from datasets import load_dataset
import re
from dotenv import load_dotenv
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import os # Added for os.getenv

from llms_api import query_model


EIGHT_SHOT_EXAMPLES = """
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

MATH_SHOT_EXAMPLES = """
See the examples below to guide you on how your answer format should be for MATH problems:

Q: Let $x$ and $y$ be positive real numbers such that $x+y=1$. Find the minimum value of $\frac{1}{x} + \frac{4}{y}$.
A: Let's think step by step.
We want to minimize $f(x,y) = \frac{1}{x} + \frac{4}{y}$ subject to $x+y=1$ and $x,y > 0$.
Since $y=1-x$, we can rewrite $f(x) = \frac{1}{x} + \frac{4}{1-x}$.
To find the minimum, we can take the derivative with respect to $x$ and set it to 0.
$f'(x) = -\frac{1}{x^2} - \frac{4(-1)}{(1-x)^2} = -\frac{1}{x^2} + \frac{4}{(1-x)^2}$.
Set $f'(x)=0$: $\frac{4}{(1-x)^2} = \frac{1}{x^2}$.
This implies $4x^2 = (1-x)^2$.
Taking the square root of both sides, $2x = 1-x$ (since $x, 1-x > 0$).
So $3x = 1$, which means $x = \frac{1}{3}$.
Then $y = 1 - x = 1 - \frac{1}{3} = \frac{2}{3}$.
The minimum value is $\frac{1}{1/3} + \frac{4}{2/3} = 3 + \frac{12}{2} = 3 + 6 = 9$.
To confirm it's a minimum, we check the second derivative:
$f''(x) = \frac{2}{x^3} + \frac{8}{(1-x)^3}$. For $x \in (0,1)$, $f''(x) > 0$, so it is a minimum.
The final answer is \\boxed{9}

Q: What is the coefficient of $x^3$ in the expansion of $(2x - 3)^5$?
A: Let's think step by step.
We use the binomial theorem, which states that $(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$.
In this case, $a = 2x$, $b = -3$, and $n = 5$.
We want the term with $x^3$. Let the term be $\binom{5}{k} (2x)^{k} (-3)^{5-k}$. We want $x^k$ to be $x^3$, so $k=3$.
The term is $\binom{5}{3} (2x)^3 (-3)^{5-3}$.
$\binom{5}{3} = \frac{5!}{3!2!} = \frac{5 \times 4}{2 \times 1} = 10$.
$(2x)^3 = 2^3 x^3 = 8x^3$.
$(-3)^{5-3} = (-3)^2 = 9$.
So the term is $10 \times (8x^3) \times 9 = 720x^3$.
The coefficient of $x^3$ is $720$.
The final answer is \\boxed{720}
"""

AIME_SHOT_EXAMPLES = """
See the examples below to guide you on how your answer format should be for AIME problems:

Q: Find the smallest positive integer $n$ such that $n^2 + 20n + 19$ is a perfect square.
A: Let's think step by step.
Let $n^2 + 20n + 19 = k^2$ for some integer $k > 0$.
We complete the square: $n^2 + 20n + 19 = (n+10)^2 - 100 + 19 = (n+10)^2 - 81$.
So, $(n+10)^2 - 81 = k^2$, which means $(n+10)^2 - k^2 = 81$.
This is a difference of squares: $((n+10) - k)((n+10) + k) = 81$.
Let $A = (n+10) - k$ and $B = (n+10) + k$. So $AB = 81$.
Since $n>0$ and $k>0$ (as $(n+10)^2-81=k^2>0 \Rightarrow n+10>9$), $B > A$.
Also, $A$ and $B$ must be of the same parity. Since $AB=81$ (odd), both $A$ and $B$ must be odd.
Possible pairs $(A,B)$ with $A<B$:
1. $A=1, B=81$.
   $2(n+10) = A+B = 82 \implies n+10 = 41 \implies n=31$.
   $2k = B-A = 80 \implies k=40$. This is a valid solution.
2. $A=3, B=27$.
   $2(n+10) = A+B = 30 \implies n+10 = 15 \implies n=5$.
   $2k = B-A = 24 \implies k=12$. This is a valid solution.
The smallest positive integer $n$ is $5$.
The final answer is 5

Q: Let $N=1234567891011...4344$ be the integer obtained by writing the integers from 1 to 44 in order, one after the other. What is the remainder when $N$ is divided by 45?
A: Let's think step by step.
We need $N \pmod{45}$. Since $45 = 5 \times 9$, we find $N \pmod 5$ and $N \pmod 9$.
$N \pmod 5$: The last digit of $N$ is 4 (from 44). So $N \equiv 4 \pmod 5$.
$N \pmod 9$: $N$ is congruent to the sum of its digits modulo 9.
The sum of digits from 1 to 9 is $45 \equiv 0 \pmod 9$.
For 10 to 19: Sum of digits is $(1 \times 10) + (0+...+9) = 10+45 = 55 \equiv 1 \pmod 9$.
For 20 to 29: Sum of digits is $(2 \times 10) + (0+...+9) = 20+45 = 65 \equiv 2 \pmod 9$.
For 30 to 39: Sum of digits is $(3 \times 10) + (0+...+9) = 30+45 = 75 \equiv 3 \pmod 9$.
For 40 to 44: Digits are (4,0), (4,1), (4,2), (4,3), (4,4). Sum is $4+0+4+1+4+2+4+3+4+4 = 30 \equiv 3 \pmod 9$.
Total sum of digits $\equiv 0+1+2+3+3 \equiv 9 \equiv 0 \pmod 9$.
So $N \equiv 0 \pmod 9$.
We have $N \equiv 4 \pmod 5$ and $N \equiv 0 \pmod 9$.
$N = 9k$. So $9k \equiv 4 \pmod 5 \implies 4k \equiv 4 \pmod 5 \implies k \equiv 1 \pmod 5$.
So $k = 5m+1$. $N = 9(5m+1) = 45m+9$.
$N \equiv 9 \pmod{45}$.
The final answer is 9
"""

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
        selected_examples = MATH_SHOT_EXAMPLES
    elif benchmark_name == "AIME":
        selected_examples = AIME_SHOT_EXAMPLES
    # Default to EIGHT_SHOT_EXAMPLES for GSM types ("gsm-symbolic", "gsm8-std", "main", "p1", "p2") or any other
    else: 
        selected_examples = EIGHT_SHOT_EXAMPLES

    return f"""{selected_examples}
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

def generate_auto_reflection_auto_adapt_prompt(question, previous_incorrect_answers, auto_prompt_model_name, auto_prompt_api_key, auto_prompt_api_provider, auto_prompt_supports_sampling_params):
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
        api_provider=auto_prompt_api_provider
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
    return f"""{EIGHT_SHOT_EXAMPLES}

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
        extracted_text = re.sub(r"\\left\[", r"[", extracted_text)
        extracted_text = re.sub(r"\\right\]", r"]", extracted_text)
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
            # Return the last numerical value found as a string
            return numbers[-1]
        else:
            return None
    except Exception as e:
        print(f"Error in extract_answer_aime: {e}")
        return None