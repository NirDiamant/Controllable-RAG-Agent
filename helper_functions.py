import tiktoken
import re 
from langchain.docstore.document import Document
import PyPDF2
import pylcs
import pandas as pd
import textwrap




def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string: The input string to tokenize.
        encoding_name: The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        The number of tokens in the string according to the specified encoding.
    """

    encoding = tiktoken.encoding_for_model(encoding_name)  # Get the encoding object
    num_tokens = len(encoding.encode(string))  # Encode the string and count tokens
    return num_tokens


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text: The input text string.

    Returns:
        The text string with double newlines replaced by single newlines.
    """

    cleaned_text = re.sub(r'\n\n', '\n', text)  # Replace double newlines with single newlines
    return cleaned_text


def split_into_chapters(book_path):
    """
    Splits a PDF book into chapters based on chapter title patterns.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter number metadata.
    """

    with open(book_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        documents = pdf_reader.pages  # Get all pages from the PDF

        # Concatenate text from all pages
        text = " ".join([doc.extract_text() for doc in documents])

        # Split text into chapters based on chapter title pattern (adjust as needed)
        chapters = re.split(r'(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)', text)

        # Create Document objects with chapter metadata
        chapter_docs = []
        chapter_num = 1
        for i in range(1, len(chapters), 2):
            chapter_text = chapters[i] + chapters[i + 1]  # Combine chapter title and content
            doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
            chapter_docs.append(doc)
            chapter_num += 1

    return chapter_docs


def extract_book_quotes_as_documents(documents, min_length=50):
    quotes_as_documents = []
    # Correct pattern for quotes longer than min_length characters, including line breaks
    quote_pattern_longer_than_min_length = re.compile(rf'“(.{{{min_length},}}?)”', re.DOTALL)

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        found_quotes = quote_pattern_longer_than_min_length.findall(content)
        for quote in found_quotes:
            quote_doc = Document(page_content=quote)
            quotes_as_documents.append(quote_doc)
    
    return quotes_as_documents



def escape_quotes(text):
  """Escapes both single and double quotes in a string.

  Args:
    text: The string to escape.

  Returns:
    The string with single and double quotes escaped.
  """
  return text.replace('"', '\\"').replace("'", "\\'")



def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def is_similarity_ratio_lower_than_th(large_string, short_string, th):
    """
    Checks if the similarity ratio between two strings is lower than a given threshold.

    Args:
        large_string: The larger string to compare.
        short_string: The shorter string to compare.
        th: The similarity threshold.

    Returns:
        True if the similarity ratio is lower than the threshold, False otherwise.
    """

    # Calculate the length of the longest common subsequence (LCS)
    lcs = pylcs.lcs_sequence_length(large_string, short_string)

    # Calculate the similarity ratio
    similarity_ratio = lcs / len(short_string)

    # Check if the similarity ratio is lower than the threshold
    if similarity_ratio < th:
        return True
    else:
        return False
    

def analyse_metric_results(results_df):
    """
    Analyzes and prints the results of various metrics.

    Args:
        results_df: A pandas DataFrame containing the metric results.
    """

    for metric_name, metric_value in results_df.items():
        print(f"\n**{metric_name.upper()}**")

        # Extract the numerical value from the Series object
        if isinstance(metric_value, pd.Series):
            metric_value = metric_value.values[0]  # Assuming the value is at index 0

        # Print explanation and score for each metric
        if metric_name == "faithfulness":
            print("Measures how well the generated answer is supported by the retrieved documents.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better faithfulness.
        elif metric_name == "answer_relevancy":
            print("Measures how relevant the generated answer is to the question.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better relevance.
        elif metric_name == "context_precision":
            print("Measures the proportion of retrieved documents that are actually relevant.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better precision (avoiding irrelevant documents).
        elif metric_name == "context_relevancy":
            print("Measures how relevant the retrieved documents are to the question.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better relevance of retrieved documents.
        elif metric_name == "context_recall":
            print("Measures the proportion of relevant documents that are successfully retrieved.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better recall (finding all relevant documents).
        elif metric_name == "context_entity_recall":
            print("Measures the proportion of relevant entities mentioned in the question that are also found in the retrieved documents.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better recall of relevant entities.
        elif metric_name == "answer_similarity":
            print("Measures the semantic similarity between the generated answer and the ground truth answer.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates closer semantic meaning between the answers.
        elif metric_name == "answer_correctness":
            print("Measures whether the generated answer is factually correct.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better correctness.



import dill

def save_object(obj, filename):
    """
    Save a Python object to a file using dill.
    
    Args:
    - obj: The Python object to save.
    - filename: The name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        dill.dump(obj, file)
    print(f"Object has been saved to '{filename}'.")

def load_object(filename):
    """
    Load a Python object from a file using dill.
    
    Args:
    - filename: The name of the file from which the object will be loaded.
    
    Returns:
    - The loaded Python object.
    """
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    print(f"Object has been loaded from '{filename}'.")
    return obj

# Example usage:
# save_object(plan_and_execute_app, 'plan_and_execute_app.pkl')
# plan_and_execute_app = load_object('plan_and_execute_app.pkl')

