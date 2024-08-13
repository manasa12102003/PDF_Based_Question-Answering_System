#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install dependencies
get_ipython().system('pip install transformers')


# In[2]:


get_ipython().system('pip install PyPDF2')


# In[3]:


get_ipython().system('pip install torch')


# In[4]:


#Import required packages
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import numpy as np


# In[5]:


import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


# In[6]:


from PyPDF2 import PdfReader
import sys


# In[7]:


# create bert model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# define tokenizer for bert
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[8]:


def bert_qa(question, context, max_len=500):

    #Tokenize input question and passage 
    #Add special tokens - [CLS] and [SEP]
    input_ids = tokenizer.encode (question, context,  max_length= max_len, truncation=True)  

    #Getting number of tokens in question and context passage that contains the answer
    sep_index = input_ids.index(102) 
    len_question = sep_index + 1   
    len_context = len(input_ids)- len_question  
    
    #Separate question and context 
    #Segment ids will be 0 for question and 1 for context
    segment_ids =  [0]*len_question + [1]*(len_context)  
    
    #Converting token ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids) 

    #Getting start and end scores for answer
    #Converting input arrays to torch tensors before passing to the model
    start_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[0]
    end_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]) )[1]

    #Converting scores tensors to numpy arrays
    start_token_scores = start_token_scores.detach().numpy().flatten()
    end_token_scores = end_token_scores.detach().numpy().flatten()

    #Getting start and end index of answer based on highest scores
    answer_start_index = np.argmax(start_token_scores)
    answer_end_index = np.argmax(end_token_scores)

    #Getting scores for start and end token of the answer
    start_token_score = np.round(start_token_scores[answer_start_index], 2)
    end_token_score = np.round(end_token_scores[answer_end_index], 2)

    #Combining subwords starting with ## and get full words in output. 
    #It is because tokenizer breaks words which are not in its vocab.
    answer = tokens[answer_start_index] 
    for i in range(answer_start_index + 1, answer_end_index + 1):
        if tokens[i][0:2] == '##':  
            answer += tokens[i][2:] 
        else:
            answer += ' ' + tokens[i]  

    # If the answer not in the passage
    if ( answer_start_index == 0) or (start_token_score < 0 ) or  (answer == '[SEP]') or ( answer_end_index <  answer_start_index):
        answer = "Sorry, Couldn't find answer in given pdf. Please try again!"
    
    return (answer_start_index, answer_end_index, start_token_score, end_token_score,  answer)


# In[9]:


import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PyPDF2 import PdfReader

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

def get_answer():
    file_path = file_entry.get()
    question = question_entry.get()
    
    if not file_path or not question:
        messagebox.showwarning("Input Error", "Please select a PDF file and enter a question.")
        return
    
    pdf_reader = PdfReader(open(file_path, 'rb'))
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    answer = bert_qa(question, text)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, answer)

def on_closing():
    root.quit()
    root.destroy()

# Create the main Tkinter window
root = tk.Tk()
root.title("Question-Answering System using BERT")
root.configure(bg="#34495E")  # Dark blue-gray background
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create and place widgets
tk.Label(root, text="Select a PDF file:", bg="#34495E", fg="#ECF0F1", font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=5, sticky="W")
file_entry = tk.Entry(root, width=50, bg="#ECF0F1", fg="#34495E", font=("Arial", 10))
file_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_file, bg="#1ABC9C", fg="#FFFFFF", font=("Arial", 10)).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Enter question:", bg="#34495E", fg="#ECF0F1", font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=5, sticky="W")
question_entry = tk.Entry(root, width=50, bg="#ECF0F1", fg="#34495E", font=("Arial", 10))
question_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=5)

tk.Button(root, text="Submit", command=get_answer, bg="#1ABC9C", fg="#FFFFFF", font=("Arial", 10)).grid(row=2, column=1, columnspan=2, padx=10, pady=10)

tk.Label(root, text="Answer:", bg="#34495E", fg="#ECF0F1", font=("Arial", 10)).grid(row=3, column=0, padx=10, pady=5, sticky="W")
output_text = scrolledtext.ScrolledText(root, width=60, height=10, font=("Arial", 10), bg="#2C3E50", fg="#ECF0F1")
output_text.grid(row=3, column=1, columnspan=2, padx=10, pady=5)

# Start the Tkinter event loop
root.mainloop()


# In[ ]:




