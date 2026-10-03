import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("Where is API key")

client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"
role="user"
#structure it
from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str
    phonenumber:str

schema=Ticket.model_json_schema()
response_format={
    "type":"json_object"
}
system_prompt=f"""
Extract the personal information and issue from the ticket based on this schema and give a json output.
{schema}"""
message_system= {
    "role":"system",
    "content":system_prompt
}
text="Hello my name is Atul Prakash.I purchased an iphone from your store and its not working my address is delhi.My email is atulkashyap185.ak@gmail.com. My contact number is 9801288222 and my crush name is Anne Hathaway."

prompt=f""" This is a customer ticket please extract a personal information from this
{text}
"""
message={
    "role":role,
    "content":prompt
}
messages=[message_system,message]
response=client.chat.completions.create(model=model,messages=messages,response_format=response_format,temperature=2)

answer=response.choices[0].message.content

print(answer)

# How to read

import json
raw_json=answer
data_file=json.loads(raw_json)
ticket=Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)
print(ticket.phonenumber)

# ------------------------This project was about extracting personal info from the ticket---------------------------------