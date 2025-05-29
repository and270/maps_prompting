import pandas as pd
from datasets import load_dataset
import re
from dotenv import load_dotenv
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

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

SWE_BENCH_SHOT_EXAMPLES = """
Here are examples of how to respond to SWE-bench tasks:

Problem: The script `user_management.py` fails with a `KeyError` if the user's profile is missing the 'email' field. We need to handle this gracefully by defaulting to a placeholder email.

Plan:
1. Locate the section in `user_management.py` where the 'email' field is accessed.
2. Use the `.get()` method for dictionaries, providing a default value like 'no-email@example.com'.
3. Add a log message when a default email is used for better traceability.

Patch:
```diff
--- a/user_management.py
+++ b/user_management.py
@@ -25,7 +25,10 @@
 class User:
     def __init__(self, profile_data):
         self.name = profile_data['name']
-        self.email = profile_data['email']
+        self.email = profile_data.get('email', 'no-email@example.com')
+        if self.email == 'no-email@example.com':
+            # Consider logging this instead of printing for production code
+            print(f"Warning: User {self.name} has no email, using default.")
         self.user_id = profile_data['id']

     def get_details(self):
```

Problem: The date formatting function `format_date` in `utils/time_helpers.py` does not handle timezone conversions correctly, leading to off-by-one day errors for users in different timezones. The function should always convert dates to UTC before formatting.

Plan:
1. Import the `pytz` library for timezone handling.
2. Modify the `format_date` function:
    a. If the input `datetime` object is naive, assume it's UTC.
    b. Convert the timezone-aware or UTC-assumed datetime object to UTC.
    c. Format the UTC datetime object.
3. Add test cases for various input timezones.

Patch:
```diff
--- a/utils/time_helpers.py
+++ b/utils/time_helpers.py
@@ -1,5 +1,6 @@
 import datetime
+import pytz # Ensure pytz is added to requirements.txt

 def format_date(dt, fmt="%Y-%m-%d %H:%M:%S"):
     """
@@ -7,5 +8,9 @@
     If the datetime object is naive, it's assumed to be in UTC.
     """
     if dt.tzinfo is None:
-        dt = dt.replace(tzinfo=datetime.timezone.utc)
-    return dt.strftime(fmt)
+        # Assume naive datetime is UTC
+        dt_utc = pytz.utc.localize(dt)
+    else:
+        # Convert timezone-aware datetime to UTC
+        dt_utc = dt.astimezone(pytz.utc)
+    return dt_utc.strftime(fmt)
```
"""

# Meta-prompt for generating reflection prompts for MATH-like problems
MATH_META_PROMPT = """You are an expert in adapting instructions for language models. Your task is to create a personalized Self-Reflection prompt for a model that is trying to solve a mathematical problem. You will receive the original question and should adapt the prompt based on it.

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

# Meta-prompt for generating reflection prompts for SWE-bench problems
SWE_BENCH_META_PROMPT = """You are an expert in adapting instructions for language models. Your task is to create a personalized Self-Reflection prompt for a model that is trying to **fix a software bug or implement a feature described in a GitHub issue**. You will receive the original issue description and should adapt the prompt based on it.

Your task is to modify the Self-Reflection template so that it is as specific and helpful as possible for the software engineering problem. Focus on aspects such as:

*   **Type of problem**: e.g., "bug fix in data validation," "refactoring for performance," "UI layout adjustment," "incorrect API usage," "concurrency issue."
*   **Common mistakes**: e.g., "off-by-one errors," "null pointer exceptions," "race conditions," "incorrect assumptions about external APIs," "missing edge case handling," "security vulnerabilities."
*   **Complexity of the problem**: e.g., "scope of changes (single vs. multi-file)," "algorithmic complexity of the fix," "interaction with other system components," "need for new dependencies."

Here is the original Self-Reflection template that you should adapt:

--- Beginning of the template ---
You are an expert in <PROBLEM_AREA, e.g., Python bug fixing, frontend development>.
You have provided an incorrect code patch for the following issue.
Your task is to reflect on the problem, your proposed solution, and why it might have failed (e.g., based on test results or logical review).
You will then use this information to help you generate a corrected patch.
First, explain why your previous patch might have been incorrect or incomplete.
Second, list keywords that describe the type of your errors or omissions (e.g., "Logic error," "Missing edge case," "Incorrect API usage," "Concurrency").
Third, outline a plan to generate a corrected patch, step-by-step.
Fourth, create a list of detailed instructions or checks for yourself to ensure the new patch is correct.
Finally, create a list of general advice to help you solve similar types of software issues in the future.
Be concise in your response; however, capture all essential information.
For guidance, I will provide you with a single generic example problem and reflection (below).
[Example Input]
Issue: "The application crashes with a `KeyError: 'user_id'` when processing webhook events if the `user_id` field is missing from the incoming JSON payload. This happens because the code directly accesses `payload['user_id']` without checking for its existence."
Wrong Patch:
```diff
--- a/main.py
+++ b/main.py
@@ -10,6 +10,7 @@
 def process_event(payload):
   # user_id = payload['user_id'] # Original failing line
   user_id = payload.get('user_id', 'default_user') # Attempted fix
+  # Problem: 'default_user' might not be acceptable for all downstream processes.
   print(f"Processing event for {user_id}")
   # ... further processing ...
```
---
[Example Output]
Explanation:
My previous patch used `payload.get('user_id', 'default_user')`. While this prevents the `KeyError`, assigning a 'default_user' might hide the actual problem or lead to incorrect data association. The core issue is that a missing `user_id` might signify an invalid event that should be logged as an error or handled differently, rather than processed with a default. The reflection should consider if the event can be processed without a user_id or if it should be rejected.
Error Keywords:
- Incorrect error handling
- Faulty assumption
- Masking underlying issue
- Input validation
Instructions:
1. Re-evaluate the requirement: Is `user_id` absolutely mandatory?
2. If mandatory, the patch should reject payloads missing `user_id` or log a critical error.
3. If optional, ensure downstream components correctly handle a `None` or special marker for `user_id`.
4. Do not introduce placeholder values that could corrupt data or hide errors.
5. Consider the impact on data integrity and system behavior.
Advice:
- Always understand the full context of an error before patching.
- Prefer failing fast or explicitly handling missing data over introducing potentially problematic defaults.
- Consider all implications of a fix, including data integrity and security.
- Ensure error messages are clear and actionable.
Corrected Plan Outline:
1. Check if 'user_id' is in payload.
2. If not, log an error and return an appropriate error response (e.g., HTTP 400 if it\'s a web handler).
3. If present, proceed with existing logic.
--- End of the template ---

Now, adapt the above template for the following issue:

Issue: {question}

Generate the adapted Self-Reflection prompt (remember, you need to create a similar example issue, a wrong patch, and the correct reflection logic as guided by the template above):
"""


def generate_cot_prompt(question, benchmark_name):
    if benchmark_name == "MATH":
        selected_examples = MATH_SHOT_EXAMPLES
    elif benchmark_name == "AIME":
        selected_examples = AIME_SHOT_EXAMPLES
    elif benchmark_name == "SWE-bench":
        selected_examples = SWE_BENCH_SHOT_EXAMPLES
    # Default to EIGHT_SHOT_EXAMPLES for GSM types ("gsm-symbolic", "gsm8-std", "main", "p1", "p2") or any other
    else: 
        selected_examples = EIGHT_SHOT_EXAMPLES

    if benchmark_name == "SWE-bench":
        # For SWE-bench, the prompt structure is different after the examples.
        return f"""{selected_examples}

Problem: {question}

Plan:""" # The LLM is expected to continue with the plan and then the patch.
    else:
        # For MATH, AIME, and GSM-like benchmarks
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

def generate_auto_reflection_auto_adapt_prompt(question, previous_incorrect_answers, auto_prompt_model, api_key, api_provider, benchmark_name):
    if benchmark_name == "SWE-bench":
        meta_prompt_template_to_use = SWE_BENCH_META_PROMPT
    else: # Default to MATH/general version, including AIME, GSM8K etc.
        meta_prompt_template_to_use = MATH_META_PROMPT
    
    current_meta_prompt = meta_prompt_template_to_use.format(question=question)
    
    adapted_reflection_prompt = query_model(
        api_key=api_key,
        prompt=current_meta_prompt, # Use the formatted prompt
        model=auto_prompt_model,
        supports_sampling_params=True, # Assuming this is still desired
        api_provider=api_provider
    )
    adapted_reflection_prompt += f"""\n\nNow, look at this question:
Question: {question}
Your initial Chain-of-Thought answer was: {previous_incorrect_answers[0] if previous_incorrect_answers else 'None'}"""
    if len(previous_incorrect_answers) > 1:
        adapted_reflection_prompt += "\n\nYour previous reflection answers were:"
        for i, answer in enumerate(previous_incorrect_answers[1:], 1):
            adapted_reflection_prompt += f"\nReflection {i}: {answer}"
    adapted_reflection_prompt += """\nYou previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections. Your new answer MUST be different from your previous answers cause they were all incorrect."""
    return adapted_reflection_prompt

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
    Extracts the content from the last \\boxed{...} block in the response text
    and applies basic LaTeX normalization.
    """
    try:
        if response_text is None:
            return None
        
        matches = re.findall(r"\\boxed{(.*?)}", response_text)
        if not matches:
            # Fallback: If no \boxed{}, try to find the last line that might contain a numerical answer or simple expression.
            # This is a heuristic and might need refinement.
            # Look for lines that are not part of "Let's think step by step" or "The final answer is"
            # and seem to contain typical answer patterns.
            lines = response_text.strip().split('\n')
            potential_answer_line = ""
            for line in reversed(lines):
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("Let's think step by step") and not clean_line.lower().startswith("the final answer is"):
                    # Heuristic: if it contains numbers and/or some math symbols, consider it.
                    if re.search(r'[\d\.\/\*\-\+\(\)pi=sqrt]', clean_line) or re.search(r'\\[a-zA-Z]+', clean_line) :
                        potential_answer_line = clean_line
                        break # Take the first such line from the bottom
            if not potential_answer_line:
                return None # No suitable fallback line found
            extracted_text = potential_answer_line
        else:
            extracted_text = matches[-1]

        # Basic LaTeX Normalizations
        # Remove \text{...}, \mathrm{...} if they wrap the whole expression or are simple
        extracted_text = re.sub(r"\\text{\s*(.*?)\s*}", r"\1", extracted_text)
        extracted_text = re.sub(r"\\mathrm{\s*(.*?)\s*}", r"\1", extracted_text)

        # Spacing commands
        extracted_text = extracted_text.replace(r"\,", "")
        extracted_text = extracted_text.replace(r"\!", "")
        extracted_text = extracted_text.replace(r"\quad", " ")
        extracted_text = extracted_text.replace(r"\qquad", "  ")
        
        # Common symbols and functions
        extracted_text = extracted_text.replace(r"\pi", "pi")
        # More robust \frac replacement
        extracted_text = re.sub(r"\\frac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}", r"(\1)/(\2)", extracted_text)
        extracted_text = re.sub(r"\\sqrt\s*{\s*(.*?)\s*}", r"sqrt(\1)", extracted_text)
        extracted_text = extracted_text.replace(r"\cdot", "*")
        extracted_text = extracted_text.replace(r"\times", "*")
        extracted_text = extracted_text.replace(r"^{\circ}", "") # remove degree symbol

        # Remove unnecessary outer braces: e.g., {{expr}} -> {expr}
        # This can be tricky; a simple pass for now.
        if extracted_text.startswith("{") and extracted_text.endswith("}"):
            # Check if braces are balanced and truly just wrapping
            temp_text = extracted_text[1:-1]
            open_braces = 0
            balanced = True
            for char in temp_text:
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                if open_braces < 0: # Closing brace before matching open
                    balanced = False
                    break
            if balanced and open_braces == 0 : # Ensure all internal braces are matched
                 extracted_text = temp_text
        
        # Remove dollar signs if they are just wrapping the expression
        if extracted_text.startswith("$") and extracted_text.endswith("$"):
            extracted_text = extracted_text[1:-1]

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


def extract_patch_swe_bench(response_text):
    """
    Extracts a code patch from the response text, typically enclosed in
    triple backticks with an optional "diff" language specifier.
    e.g.:
    ```diff
    --- a/file.py
    +++ b/file.py
    @@ -1,1 +1,1 @@
    -old line
    +new line
    ```
    Returns the content of the first such block found, or None.
    """
    try:
        if response_text is None:
            return None

        # Primary regex: attempts to match ```diff\n...content...\n``` or ```\n...content...\n```
        match_specific = re.search(r"```(?:diff)?\n(.*?)\n```", response_text, re.DOTALL)
        
        if match_specific:
            patch = match_specific.group(1)
            return patch.strip()
        else:
            # Fallback regex: captures content within the first pair of triple backticks.
            match_general = re.search(r"```(.*?)```", response_text, re.DOTALL)
            if match_general:
                patch_content = match_general.group(1)
                
                lines = patch_content.split('\n', 1)
                common_specifiers = [
                    "diff", "python", "text", "patch", "sh", "bash", "javascript", 
                    "html", "css", "json", "yaml", "xml", "sql", "java", "c", "cpp", 
                    "csharp", "php", "ruby", "perl", "go", "rust", "swift", "kotlin", 
                    "typescript", "" 
                ]
                
                if len(lines) > 1 and lines[0].strip().lower() in common_specifiers:
                    patch = lines[1] 
                elif len(lines) == 1 and lines[0].strip().lower() in common_specifiers:
                    patch = "" 
                else:
                    patch = patch_content 
                
                return patch.strip()
            return None 
            
    except Exception as e:
        print(f"Error in extract_patch_swe_bench: {e}. Response text (first 1000 chars): '{str(response_text)[:1000]}'")
        return None
