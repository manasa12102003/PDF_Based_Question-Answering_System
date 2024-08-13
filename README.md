# PDF_Based_Question_Answering_System

This project is an innovative Question-Answering System powered by BERT (Bi-Directional Encoder Representations from Transformers), seamlessly integrated with a user-friendly Tkinter interface. The system allows users to effortlessly browse and select any PDF file, pose natural language questions, and receive precise answers extracted from the document. By leveraging BERT's advanced natural language processing capabilities, the system understands the context and nuance of the questions, providing highly relevant answers in real-time. This tool exemplifies the future of document interaction, making information retrieval from lengthy PDFs as simple as browsing, asking, and discovering—right at your fingertips.

## Approach : Question Answering System with Fine-Tuned BERT Technique

## Required Packages and Dependencies

This project relies on several key Python packages and libraries, each playing a crucial role in the implementation of the PDF-based Question-Answering System:

*1. PyTorch (torch):*

PyTorch is a deep learning framework that allows for easy and efficient implementation of neural networks. It is used here to run the BERT model, handling tensor operations and model inference.

*2. Hugging Face Transformers (transformers):*

The BertForQuestionAnswering class from the transformers library is the core of the question-answering system. It provides a pre-trained BERT model specifically fine-tuned for QA tasks.

*3. NumPy (numpy):*

NumPy is used for numerical operations, such as converting PyTorch tensors into numpy arrays. This is useful for manipulating model outputs and performing operations like finding the maximum start and end scores for answer extraction.

*4. Tkinter (tkinter, filedialog, messagebox, scrolledtext):*

Tkinter is the standard Python interface for creating graphical user interfaces (GUIs). In this project, it is used to build the user-friendly interface that allows users to select PDF files, enter questions, and view answers.

*5. PyPDF2 (PyPDF2.PdfReader):*

PyPDF2 is a library for reading PDF files. It is utilized to extract text content from the selected PDF file, which is then fed into the BERT model for question-answering.

*6. Sys (sys):*

The sys module is used to handle system-specific parameters and functions. In this project, it aids in gracefully exiting the application.

## Implementation Details: BERT-based Question Answering Function

At the core of this project is the bert_qa function, which is responsible for extracting the most relevant answers from the provided PDF content using the BERT model. Below is a detailed breakdown of how this function works:

*1. Tokenization and Input Preparation:*

The function begins by tokenizing both the input question and the context (the text extracted from the PDF). Special tokens [CLS] (used to denote the start of a sequence) and [SEP] (used to separate the question from the context) are added to the input.
The input sequence is truncated to ensure it doesn't exceed the specified maximum length (max_len), which is typically set to 500 tokens.

*2. Segment Identification:*

The function then determines the number of tokens in the question and the context, distinguishing between them by assigning segment IDs. Segment IDs are 0 for the question and 1 for the context, helping BERT to differentiate between the two parts of the input.

*3. Score Calculation:*

The function passes the tokenized input through the BERT model to obtain start and end token scores, which indicate the likelihood of each token being the beginning or end of the answer. These scores are then converted to numpy arrays for further processing.

*4. Answer Extraction:*

The function identifies the most likely start and end positions of the answer by finding the tokens with the highest start and end scores, respectively.
It then reconstructs the answer from these tokens. If the tokenizer has broken down any words into subwords (identified by ##), these subwords are combined to form complete words in the final answer.

*5. Handling Edge Cases:*

If the function determines that the answer cannot be found within the context (based on certain conditions, such as low scores or the presence of special tokens), it returns a default message indicating that the answer could not be found in the PDF.
This function is integral to the system's ability to deliver precise and contextually relevant answers. It ensures that despite BERT's limitations, the answers provided are as accurate and coherent as possible, enhancing the overall user experience.
