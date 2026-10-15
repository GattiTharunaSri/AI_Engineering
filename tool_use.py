import json
import display_functions
from dotenv import load_dotenv
import requests
import qrcode
from qrcode.image.styledpil import StyledPilImage
from datetime import datetime
_ =load_dotenv()

import aisuite as ai
client = ai.Client()

def get_current_time():
    """Returns the current time as a string"""
    return datetime.now().strftime("%H:%M:%S")

time_ = get_current_time()
print(time_)

prompt = "What time is it?"
messages = [{
    "role": "user",
    "content" : prompt
}]

response = client.chat.completions.create(
    model = "openai:gpt-4o",
    messages=messages,
    tools = [get_current_time],
    max_turns = 5
)
#print(response.choices[0].message.content)
display_functions.pretty_print_chat_completion(response)

tools = [
    {
        "type": "function",
        "function":{
            "name" : "get_current_time",
            "description": "Returns the current time as a string.",
            "parameters":{}
        }
    }
]
response1 = client.chat.completions.create(
    model= "openai:gpt-4o",
    messages=messages,
    tools = tools
)
print(json.dumps(response1.model_dump(),indent= 2, default = str))

response2 = None
if response1.choices[0].message.tool_calls:
    tool_call = response1.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    tool_result = get_current_time()
    messages.append(response1.choices[0].message)
    messages.append({
        "role" : "tool",
        "tool_call_id":tool_call.id,
        "content": str(tool_result)
    })
    response2 = client.chat.completions.create(
        model = "openai:gpt-4o",
        messages = messages,
        tools = tools
    )
    print(response2.choices[0].message.content)

