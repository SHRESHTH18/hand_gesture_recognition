from api_key import get_gemini_key
from IPython.display import display
from IPython.display import Markdown

import os
import google.generativeai as genai

os.environ['GOOGLE_API_KEY']=get_gemini_key()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro',
                              generation_config={
                                  "temperature":0.7,
                                  "max_output_tokens":50
                              })
def response(prompt):
    response=model.generate_content(prompt)
    text=str(response.text)
    return text
