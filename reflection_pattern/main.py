import re
import json
import utils

df = utils.load_and_prepare_data('coffee_sales.csv')
utils.print_html(df.sample(n=5))

def generate_chart_code(instruction: str, model: str, out_path_v1: str) -> str:
    prompt = f"""
    You are a Data analyst.
    Return your answer STRICTLY in this format:
    
    <execute_python>
    #valid python code here
    </execute_python>

    Do not add explanations, only the tags and the code.

    The code should create a visualization from a DataFrame 'df' with these columns:
    - date   (datetime64 — already parsed; use df['date'].dt.year, df['date'].dt.month, etc.)
    - time   (string, HH:MM — do NOT concatenate or combine with the date column)
    - cash_type (string: 'card' or 'cash')
    - card (string)
    - price (number)
    - coffee_name (string)
    - quarter (int, 1–4 — already computed, use directly)
    - month  (int, 1–12 — already computed, use directly)
    - year   (int, e.g. 2024 — already computed, use directly)

    User instruction : {instruction}

    Requirements for the code:
    1. Assume the dataframe is already loaded as 'df'.
    2. Use matplotlib for plotting.
    3. Add clear labels, titles and legend if needed.
    4. Save the figure as '{out_path_v1}' with dpi = 300
    5. Dont call plt.show().
    6. Close all plots with plt.close()
    7. Add all necessary import python statements
    8. CRITICAL: 'date' is datetime64 - nevere use string concatenation on it.
        Filter by year/quarter using the 'year' and 'quarter' integer columns.
    Return ONLY the code wrapped in <execute_python> tags.


    """
    response =utils.get_response(model,prompt)
    return response
code_v1 = generate_chart_code(
    instruction=" Create a plot comparing the Q1 sales in 2024 and 2025 using the data in coffee_sales.csv .",
    model="gpt-4o-mini",
    out_path_v1="chart_v1.png"
)

utils.print_html(code_v1,title = "LLM output with first draft code")

match  = re.search(r"<execute_python>([\s\S]*?)</execute_python>",code_v1)
if match:
    initial_code = match.group(1).strip()
    utils.print_html(initial_code, title = "Extracted code to execute")
    exec_globals = {"df":df}
    exec(initial_code,exec_globals)
utils.print_html(
    content = "chart_v1.png",
    title = "Generated chart V1",
    is_image = True
)

def reflect_and_regenerate_code(
        chart_path: str, model_name: str, instruction: str, code_v1: str, out_path_v2: str, 
)-> tuple[str,str]:
    """Crtique the chart IMAGE and the original code against the instruction, then return refined matplotlib code.
    Return (feedback, refined_code_with_tags)
    Supports OpenAI and claude models"""
    media_type, b64 = utils.encode_image_b64(chart_path)
    prompt = f"""
    You are a Data Visuavalization expert.
    Your Task: Critique the attached chart and the original code against the instruction and then return improved
    matplotlib code.
    Original Code (For context):
    {code_v1}

    OUTPUT FORMAT (STRICT):
    1. First line: a valid json object with ONLY the "feedback" field.
    Example: {{"feedback": "The legend is unclear and the axis_labels are over the bars"}}
    2. After a new line, output ONLY the refined python code wrapped in <execute_python> ... </execute_python> tags.
    3. Import all the necessary libraries in the code. Dont assume any imports from previous code.

    HARD CONSTRAINTS:
    - Do NOT include Markdown, Backticks, or any extra prose outside the two parts above.
    - Use Pandas/matplotlib only (no seaborn).
    - Assume df already exists; donot read from files.
    - Save to '{out_path_v2}'with dpi = 300.
    - Always call plt.close() at the end (no plt.show()).
    - Include all necessary import statements.
    
    IMPORTANT: The 'date' column is already a pandas datetime64 type.
    - Do NOT concatenate 'date' with 'time' using string operations.
    - To filter by year/quarter, use: df[df['year'] == 2024] or df['date'].dt.year == 2024
    - The 'quarter' and 'year' columns already exist as integers; use them directly.

    Schema (columns available in df):
    - date   (datetime64 — already parsed; use df['date'].dt.year, etc.)
    - time   (string, HH:MM — do NOT concatenate with date)
    - cash_type (string: 'card' or 'cash')
    - card   (string)
    - price  (float)
    - coffee_name (string)
    - quarter (int, 1–4)
    - month  (int, 1–12)
    - year   (int)
    
    CRITICAL TYPE RULE: 'date' is already datetime64.
    - NEVER do: df['date'] + ' ' + df['time']  ← this will crash
    - ALWAYS filter by year/quarter using the integer columns: df[df['year'] == 2024]

    Instruction:
    {instruction}
    """
    lower = model_name.lower()
    if "claude" in lower or "anthropic" in lower:
        content = utils.image_anthropic_call(model_name,prompt,media_type, b64)
    else:
        content = utils.image_openai_call(model_name,prompt,media_type,b64)
    lines = content.strip().splitlines()
    json_line = lines[0].strip() if lines else ""

    try:
        obj = json.loads(json_line)
    except Exception as e:
        m_json = re.search(r"\{.*?\}",content, flags = re.DOTALL)
        if m_json:
            try:
                obj = json.loads(m_json.group(0))
            except Exception as e2:
                obj = {"feedback": f"Failed to parse JSON: {e2}", "refined_code": ""}
        else:
            obj = {"feedback": f"Failed to find JSON: {e}", "refined_code" : ""}
    m_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>",content)
    refined_code_body = m_code.group(1).strip() if m_code else ""
    refined_code =utils.ensure_execute_python_tags(refined_code_body)
    feedback = str(obj.get("feedback","")).strip()
    return feedback, refined_code

feedback, code_v2 = reflect_and_regenerate_code(
    chart_path="C:/Users/tharu/Downloads/agentic_AI/reflection_pattern/chart_v1.png",
    model_name="o4-mini",
    instruction="Create a plot compaing Q1 coffee sales in 2024 and 2025 using the data in cofee_sales.csv",
    out_path_v2="chart_v2.png",
    code_v1=code_v1
)
utils.print_html(feedback, title = "Feedback on V1 Chart")
utils.print_html(code_v2, title = "Regenerated Code Output V2")

match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v2)
if match:
    reflected_code = match.group(1).strip()
    exec_globals = {"df": df}
    exec(reflected_code, exec_globals)

# If code run successfully, the file chart_v2.png should have been generated
utils.print_html(
    content="chart_v2.png",
    title="Regenerated Chart (V2)",
    is_image=True
)