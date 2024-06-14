from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq
from langchain.vectorstores import  FAISS
from langchain.embeddings import OpenAIEmbeddings 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pprint import pprint
import os
from typing_extensions import TypedDict
from typing import List, TypedDict


import streamlit as st
from pyvis.network import Network
import tempfile
import langgraph


### Helper functions for the notebook
from helper_functions import escape_quotes, text_wrap

load_dotenv()

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')








embeddings = OpenAIEmbeddings()
chunks_vector_store =  FAISS.load_local("chunks_vector_store", embeddings, allow_dangerous_deserialization=True)
chapter_summaries_vector_store =  FAISS.load_local("chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True)


chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})     
chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})

def retrieve_context_per_question(state):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    # Retrieve relevant documents
    print("Retrieving relevant chunks...")
    question = state["question"]
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    context = " ".join(doc.page_content for doc in docs)



    print("Retrieving relevant chapter summaries...")
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(state["question"])

    # Concatenate chapter summaries with citation information
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )


    all_contexts = context + context_summaries
    all_contexts = escape_quotes(all_contexts)

    return {"context": all_contexts, "question": question}



keep_only_relevant_content_prompt_template = """you receive a query: {query} and retrieved docuemnts: {retrieved_documents} from a
 vector store.
 You need to filter out all the non relevant information that don't supply important information regarding the {query}.
 your goal is just to filter out the non relevant information.
 you can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
 DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
 output the filtered relevant content.
"""


class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(description="The relevant content from the retrieved documents that is relevant to the query.")


keep_only_relevant_content_json_parser = JsonOutputParser(pydantic_object=KeepRelevantContent)

keep_only_relevant_content_prompt = PromptTemplate(
    template=keep_only_relevant_content_prompt_template,
    input_variables=["query", "retrieved_documents"],
)


keep_only_relevant_content_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
keep_only_relevant_content_chain = keep_only_relevant_content_prompt | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)


def keep_only_relevant_content(state):
    """
    Keeps only the relevant content from the retrieved documents that is relevant to the query.

    Args:
        question: The query question.
        context: The retrieved documents.
        chain: The LLMChain instance.

    Returns:
        The relevant content from the retrieved documents that is relevant to the query.
    """
    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "retrieved_documents": context
}
    print("keeping only the relevant content...")
    pprint("--------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {"relevant_context": relevant_content, "context": context, "question": question}



class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")

question_answer_from_context_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)


question_answer_cot_prompt_template = """ 
Examples of Chain-of-Thought Reasoning

Example 1

Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
Question: Who is the tallest person?
Reasoning Chain:
The context tells us Mary is taller than Jane
It also says Jane is shorter than Tom
And Tom is the same height as David
So the order from tallest to shortest is: Mary, Tom/David, Jane
Therefore, Mary must be the tallest person

Example 2
Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
 A third spell created a bright light at the end of the caster's wand.
Question: Based on the context, if Harry cast these spells, what could he do?
Reasoning Chain:
The context describes three different magic spells
The first spell allows turning a person into an animal temporarily
The second spell can levitate or float objects
The third spell creates a bright light
If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
Instructions.

Example 3 
Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
Question: Why did Harry receive a broomstick for his birthday?
Reasoning Chain:
The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
However, the context does not provide any information about why he received that specific present or who gave it to him.
There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
 just like in the previous examples.
Context
{context}
Question
{question}
"""

question_answer_from_context_cot_prompt = PromptTemplate(
    template=question_answer_cot_prompt_template,
    input_variables=["context", "question"],
)
question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)


def answer_question_from_context(state):
    """
    Answers a question from a given context.

    Args:
        question: The query question.
        context: The context to answer the question from.
        chain: The LLMChain instance.

    Returns:
        The answer to the question from the context.
    """
    question = state["question"]
    context = state["aggregated_context"] if "aggregated_context" in state else state["context"]

    input_data = {
    "question": question,
    "context": context
}
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f'answer before checking hallucination: {answer}')
    return {"answer": answer, "context": context, "question": question}







is_relevant_content_prompt_template = """you receive a query: {query} and a context: {context} retrieved from a vector store. 
You need to determine if the document is relevant to the query. 

{format_instructions}"""

class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")

is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)
is_relevant_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)

is_relevant_content_prompt = PromptTemplate(
    template=is_relevant_content_prompt_template,
    input_variables=["query", "context"],
    partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()},
)
is_relevant_content_chain = is_relevant_content_prompt | is_relevant_llm | is_relevant_json_parser

def is_relevant_content(state):
    """
    Determines if the document is relevant to the query.

    Args:
        question: The query question.
        context: The context to determine relevance.
    """

    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "context": context
}

    # Invoke the chain to determine if the document is relevant
    output = is_relevant_content_chain.invoke(input_data)
    print("Determining if the document is relevant...")
    if output["is_relevant"] == True:
        print("The document is relevant.")
        return "relevant"
    else:
        print("The document is not relevant.")
        return "not relevant"


class is_grounded_on_facts(BaseModel):
    """
    Output schema for the rewritten question.
    """
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

is_grounded_on_facts_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
you don't mind if it doesn't make sense, as long as it is grounded in the context.
output a json containing the answer to the question, and appart from the json format don't output any additional text.

 """
is_grounded_on_facts_prompt = PromptTemplate(
    template=is_grounded_on_facts_prompt_template,
    input_variables=["context", "answer"],
)
is_grounded_on_facts_chain = is_grounded_on_facts_prompt | is_grounded_on_facts_llm.with_structured_output(is_grounded_on_facts)


can_be_answered_prompt_template = """You receive a query: {question} and a context: {context}. 
You need to determine if the question can be fully answered based on the context.
{format_instructions}
"""

class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(description="binary result of whether the question can be fully answered or not")
    explanation: str = Field(description="An explanation of why the question can be fully answered or not.")

can_be_answered_json_parser = JsonOutputParser(pydantic_object=QuestionAnswer)

answer_question_prompt = PromptTemplate(
    template=can_be_answered_prompt_template,
    input_variables=["question","context"],
    partial_variables={"format_instructions": can_be_answered_json_parser.get_format_instructions()},
)

can_be_answered_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)
can_be_answered_chain = answer_question_prompt | can_be_answered_llm | can_be_answered_json_parser


is_distilled_content_grounded_on_content_prompt_template = """you receive some distilled content: {distilled_content} and the original context: {original_context}.
    you need to determine if the distilled content is grounded on the original context.
    if the distilled content is grounded on the original context, set the grounded field to true.
    if the distilled content is not grounded on the original context, set the grounded field to false. {format_instructions}"""
  

class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(description="Whether the distilled content is grounded on the original context.")
    explanation: str = Field(description="An explanation of why the distilled content is or is not grounded on the original context.")

is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(pydantic_object=IsDistilledContentGroundedOnContent)

is_distilled_content_grounded_on_content_prompt = PromptTemplate(
    template=is_distilled_content_grounded_on_content_prompt_template,
    input_variables=["distilled_content", "original_context"],
    partial_variables={"format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()},
)

is_distilled_content_grounded_on_content_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)

is_distilled_content_grounded_on_content_chain = is_distilled_content_grounded_on_content_prompt | is_distilled_content_grounded_on_content_llm | is_distilled_content_grounded_on_content_json_parser


def is_distilled_content_grounded_on_content(state):
    pprint("--------------------")

    """
    Determines if the distilled content is grounded on the original context.

    Args:
        distilled_content: The distilled content.
        original_context: The original context.

    Returns:
        Whether the distilled content is grounded on the original context.
    """

    print("Determining if the distilled content is grounded on the original context...")
    distilled_content = state["relevant_context"]
    original_context = state["context"]

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context
    }

    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"


class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    question: str
    context: str
    relevant_context: str

qualitative_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

# Define the nodes
qualitative_retrieval_workflow.add_node("retrieve_context_per_question",retrieve_context_per_question)
qualitative_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

# Build the graph
qualitative_retrieval_workflow.set_entry_point("retrieve_context_per_question")
qualitative_retrieval_workflow.add_edge("retrieve_context_per_question", "keep_only_relevant_content")

qualitative_retrieval_workflow.add_conditional_edges(
    "keep_only_relevant_content",
    is_distilled_content_grounded_on_content,
    {"grounded on the original context":END,
      "not grounded on the original context":"keep_only_relevant_content"},
    )


qualitative_retrieval_workflow_app = qualitative_retrieval_workflow.compile()



def is_answer_grounded_on_context(state):
    """Determines if the answer to the question is grounded in the facts.
    
    Args:
        state: A dictionary containing the context and answer.
    """
    print("Checking if the answer is grounded in the facts...")
    context = state["context"]
    answer = state["answer"]
    
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"


class QualitativeAnswerGraphState(TypedDict):
    """
    Represents the state of our graph.

    """

    question: str
    context: str
    answer: str

qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

# Define the nodes

qualitative_answer_workflow.add_node("answer_question_from_context",answer_question_from_context)

# Build the graph
qualitative_answer_workflow.set_entry_point("answer_question_from_context")

qualitative_answer_workflow.add_conditional_edges(
"answer_question_from_context",is_answer_grounded_on_context ,{"hallucination":"answer_question_from_context", "grounded on context":END}

)

qualitative_answer_workflow_app = qualitative_answer_workflow.compile()


class PlanExecute(TypedDict):
    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: dict 
    curr_context: str
    aggregated_context: str
    tool: str
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt =""" For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

"""

planner_prompt = PromptTemplate(
    template=planner_prompt,
      input_variables=["question"], 
     )

planner_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)

planner = planner_prompt | planner_llm.with_structured_output(Plan)

break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
you need to go through the plan refine it according to this:
1. every step has to be able to be executed by either retrieving relevant information from a vector store based on a given query, or answering a question from a given context.
2. every step should contain all the information needed to execute it.

output the refined plan
"""

break_down_plan_prompt = PromptTemplate(
    template=break_down_plan_prompt_template,
    input_variables=["plan"],
)

break_down_plan_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)

break_down_plan_chain = break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)


class ActPossibleResults(BaseModel):
    """Possible results of the action."""
    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")
    

act_possible_results_parser = JsonOutputParser(pydantic_object=ActPossibleResults)

replanner_prompt_template =""" For the given objective, come up with a simple step by step plan of how to figure out the answer. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

You already have the following context:
{aggregated_context}

Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
Do not return previously done steps as part of the plan.

the format is json so escape quotes and new lines.

{format_instructions}

"""

replanner_prompt = PromptTemplate(
    template=replanner_prompt_template,
    input_variables=["question", "plan", "past_steps", "aggregated_context"],
    partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
)

replanner_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)



replanner = replanner_prompt | replanner_llm | act_possible_results_parser


tasks_handler_prompt_template = """You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
You have the following tools at your disposal:
Tool A: a tool that retrieves relevant information from a vector store based on a given query.
use Tool A when you think the current task should be executed by retrieving relevant information from a vector store.
Tool B: a tool that answers a question from a given context.
use Tool B ONLY when you the current task can be answered by the aggregated context {aggregated_context}

You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
if you decide to use Tool A, output the query to be used for the tool and also that the tool to be used is Tool A.
if you decide to use Tool B, output the question to be used for the tool, the context, and also that the tool to be used is Tool B.
"""

class TaskHandlerOutput(BaseModel):
    """Output schema for the task handler."""
    query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
    curr_context: str = Field(description="The context to be based on in order to answer the query.")
    tool: str = Field(description="The tool to be used should be either 'retrieve' or 'answer_from_context'.")


task_handler_prompt = PromptTemplate(
    template=tasks_handler_prompt_template,
    input_variables=["curr_task", "aggregated_context", "past_steps", "question"],
)

task_handler_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput)


class AnonymizeQuestion(BaseModel):
    """Anonymized question and mapping."""
    anonymized_question : str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.")
    explanation: str = Field(description="Explanation of the action.")

anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)


anonymize_question_prompt_template = """ You are a question anonymizer. The input You receive is a string containing several words that
 construct a question {question}. Your goal is to changes all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
 ```example1:
        if the input is \"who is harry potter?\" the output should be \"who is X?\" and the mapping should be {{\"X\": \"harry potter\"}} ```
```example2:
        if the input is \"how did the bad guy played with the alex and rony?\"
          the output should be \"how did the X played with the Y and Z?\" and the mapping should be {{\"X\": \"bad guy\", \"Y\": \"alex\", \"Z\": \"rony\"}}```
 you must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
  output the anonymized question and the mapping in a json format. {format_instructions}"""



anonymize_question_prompt = PromptTemplate(
    template=anonymize_question_prompt_template,
    input_variables=["question"],
    partial_variables={"format_instructions": anonymize_question_parser.get_format_instructions()},
)

anonymize_question_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
anonymize_question_chain = anonymize_question_prompt | anonymize_question_llm | anonymize_question_parser



class DeAnonymizePlan(BaseModel):
    """Possible results of the action."""
    plan: List = Field(description="Plan to follow in future. with all the variables replaced with the mapped words.")

deanonymize_plan_parser = JsonOutputParser(pydantic_object=DeAnonymizePlan)


de_anonymize_plan_prompt_template = """ you receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. you also receive
the mapping for those variables to words {mapping}. replace all the variables in the list of tasks with the mapped words. if no variables are present,
return the original list of tasks. in any case, just output the updated list of tasks in a json format as described here, without any additional text apart from the
"""


de_anonymize_plan_prompt = PromptTemplate(
    template=de_anonymize_plan_prompt_template,
    input_variables=["plan", "mapping"],
)

de_anonymize_plan_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
de_anonymize_plan_chain = de_anonymize_plan_prompt | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan)

class CanBeAnsweredAlready(BaseModel):
    """Possible results of the action."""
    can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")

can_be_answered_already_prompt_template = """You receive a query: {question} and a context: {context}.
You need to determine if the question can be fully answered relying only the given context.
The only infomation you have and can rely on is the context you received. 
you have no prior knowledge of the question or the context.
if you think the question can be answered based on the context, output 'true', otherwise output 'false'.
"""

can_be_answered_json_parser = JsonOutputParser(pydantic_object=CanBeAnsweredAlready)

can_be_answered_already_prompt = PromptTemplate(
    template=can_be_answered_already_prompt_template,
    input_variables=["question","context"],
)

can_be_answered_already_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
can_be_answered_already_chain = can_be_answered_already_prompt | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)


def run_task_handler_chain(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    state["curr_state"] = "task_handler"
    print("the current plan is:")
    print(state["plan"])
    pprint("--------------------") 

    if not state['past_steps']:
        state["past_steps"] = []

    curr_task = state["plan"][0]

    inputs = {"curr_task": curr_task,
               "aggregated_context": state["aggregated_context"],
                "past_steps": state["past_steps"],
                "question": state["question"]}
    
    output = task_handler_chain.invoke(inputs)
  
    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    if output.tool == "retrieve":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="retrieve"
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"]="answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state  



def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    state["curr_state"] = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state["tool"] == "retrieve":
        return "retrieve_from_vector_store"
    elif state["tool"] == "answer":
        return "answer_from_context"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")  


def run_qualitative_retrieval_workflow(state):
    """
    Run the qualitative retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "qualitative_retrieval"
    print("Running the qualitative retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output['relevant_context']
    return state


def run_qualtative_answer_workflow(state):
    """
    Run the qualitative answer workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "qualitative_answer"
    print("Running the qualitative answer workflow...")
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["answer"]
    return state

def run_qualtative_answer_workflow_for_final_answer(state):
    """
    Run the qualitative answer workflow for the final answer.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated response.
    """
    state["curr_state"] = "qualitative_answer_final"
    print("Running the qualitative answer workflow for final answer...")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, value in output.items():
            pass  
        pprint("--------------------")
    state["response"] = value
    return state


def anonymize_queries(state: PlanExecute):
    """
    Anonymizes the question.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the anonymized question and mapping.
    """
    state["curr_state"] = "anonymize"
    print("Anonymizing question")
    pprint("--------------------")
    anonymized_question_output = anonymize_question_chain.invoke(state['question'])
    anonymized_question = anonymized_question_output["anonymized_question"]
    print(f'anonimized_querry: {anonymized_question}')
    pprint("--------------------")
    mapping = anonymized_question_output["mapping"]
    state["anonymized_question"] = anonymized_question
    state["mapping"] = mapping
    return state


def deanonymize_queries(state: PlanExecute):
    """
    De-anonymizes the plan.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the de-anonymized plan.
    """
    state["curr_state"] = "deanonymize"
    print("De-anonymizing plan")
    pprint("--------------------")
    deanonimzed_plan = de_anonymize_plan_chain.invoke({"plan": state["plan"], "mapping": state["mapping"]})
    state["plan"] = deanonimzed_plan.plan
    print(f'de-anonimized_plan: {deanonimzed_plan.plan}')
    return state


def plan_step(state: PlanExecute):
    """
    Plans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "planning"
    print("Planning step")
    pprint("--------------------")
    plan = planner.invoke({"question": state['anonymized_question']})
    state["plan"] = plan.steps
    print(f'plan: {state["plan"]}')
    return state


def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    refined_plan = break_down_plan_chain.invoke(state["plan"])
    state["plan"] = refined_plan.steps
    return state



def replan_step(state: PlanExecute):
    """
    Replans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "replanning"
    print("Replanning step")
    pprint("--------------------")
    inputs = {"question": state["question"], "plan": state["plan"], "past_steps": state["past_steps"], "aggregated_context": state["aggregated_context"]}
    output = replanner.invoke(inputs)
    state["plan"] = output['plan']['steps']
    return state


def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.
    Args:
        state: The current state of the plan execution.
    Returns:
        whether the original question can be answered or not.
    """
    state["curr_state"] = "can_be_answered"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    output = can_be_answered_already_chain.invoke(inputs)
    if output.can_be_answered == True:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        print("the aggregated context is:")
        print(text_wrap(state["aggregated_context"]))
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"



def create_agent():
    agent_workflow = StateGraph(PlanExecute)

    # Add the anonymize node
    agent_workflow.add_node("anonymize_question", anonymize_queries)

    # Add the plan node
    agent_workflow.add_node("planner", plan_step)

    # Add the break down plan node

    agent_workflow.add_node("break_down_plan", break_down_plan_step)

    # Add the deanonymize node
    agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)

    # Add the qualitative retrieval node
    agent_workflow.add_node("retrieve", run_qualitative_retrieval_workflow)

    # Add the qualitative answer node
    agent_workflow.add_node("answer", run_qualtative_answer_workflow)

    # Add the task handler node
    agent_workflow.add_node("task_handler", run_task_handler_chain)

    # Add a replan node
    agent_workflow.add_node("replan", replan_step)

    # Add answer from context node
    agent_workflow.add_node("get_final_answer", run_qualtative_answer_workflow_for_final_answer)



    # Set the entry point
    agent_workflow.set_entry_point("anonymize_question")

    # From anonymize we go to plan
    agent_workflow.add_edge("anonymize_question", "planner")

    # From plan we go to deanonymize
    agent_workflow.add_edge("planner", "de_anonymize_plan")

    # From deanonymize we go to break down plan

    agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")

    # From break_down_plan we go to task handler
    agent_workflow.add_edge("break_down_plan", "task_handler")

    # From task handler we go to either retrieve or answer
    agent_workflow.add_conditional_edges("task_handler", retrieve_or_answer, {"retrieve_from_vector_store": "retrieve", "answer_from_context": "answer"})

    # After retrieving we go to replan
    agent_workflow.add_edge("retrieve", "replan")

    # After answering we go to replan
    agent_workflow.add_edge("answer", "replan")

    # After replanning we check if the question can be answered, if yes we go to get_final_answer, if not we go to task_handler
    agent_workflow.add_conditional_edges("replan",can_be_answered, {"can_be_answered_already": "get_final_answer", "cannot_be_answered_yet": "break_down_plan"})

    # After getting the final answer we end
    agent_workflow.add_edge("get_final_answer", END)


    plan_and_execute_app = agent_workflow.compile()

    return plan_and_execute_app




# def create_network_graph(current_state):
#     net = Network(directed=True)
    
#     nodes = [
#         {"id": "anonymize_question", "label": "anonymize_question"},
#         {"id": "planner", "label": "planner"},
#         {"id": "de_anonymize_plan", "label": "de_anonymize_plan"},
#         {"id": "break_down_plan", "label": "break_down_plan"},
#         {"id": "task_handler", "label": "task_handler"},
#         {"id": "retrieve", "label": "retrieve"},
#         {"id": "answer", "label": "answer"},
#         {"id": "replan", "label": "replan"},
#         {"id": "can_be_answered_already", "label": "can_be_answered_already"},
#         {"id": "get_final_answer", "label": "get_final_answer"}
#     ]
    
#     edges = [
#         ("anonymize_question", "planner"),
#         ("planner", "de_anonymize_plan"),
#         ("de_anonymize_plan", "break_down_plan"),
#         ("break_down_plan", "task_handler"),
#         ("task_handler", "retrieve"),
#         ("task_handler", "answer"),
#         ("retrieve", "replan"),
#         ("answer", "replan"),
#         ("replan", "can_be_answered_already"),
#         ("can_be_answered_already", "get_final_answer")
#     ]
    
#     for node in nodes:
#         color = "green" if node["id"] == current_state else "pink"
#         net.add_node(node["id"], label=node["label"], color=color)
    
#     for edge in edges:
#         net.add_edge(*edge)
    
#     return net

# def save_and_display_graph(net):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
#         net.write_html(tmp_file.name)
#         return tmp_file.name

# def execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder, recursion_limit=15):
#     """
#     Execute the plan and print the steps.
#     Args:
#         inputs: The inputs to the plan.
#         recursion_limit: The recursion limit.
#     Returns:
#         The response and the final state.
#     """
#     config = {"recursion_limit": recursion_limit}
#     agent_state_value = None
#     progress_bar = st.progress(0)
#     step = 0

#     try:    
#         for plan_output in plan_and_execute_app.stream(inputs, config=config):
#             step += 1
#             for _, agent_state_value in plan_output.items():
#                 # Update placeholders
#                 for key, placeholder in placeholders.items():
#                     if key in agent_state_value and key != "curr_state":
#                         if key in ["plan", "past_steps"] and isinstance(agent_state_value[key], list):
#                             formatted_value = "\n".join([f"{i+1}. {item}" for i, item in enumerate(agent_state_value[key])])
#                         else:
#                             formatted_value = agent_state_value[key]
#                         placeholder.markdown(f"{formatted_value}")
                
#                 current_state = agent_state_value["curr_state"]
#                 net = create_network_graph(current_state)
#                 graph_filename = save_and_display_graph(net)
#                 with open(graph_filename, "r") as f:
#                     graph_html = f.read()
#                 graph_placeholder.markdown(f'<div style="height:600px;">{graph_html}</div>', unsafe_allow_html=True)
                
#                 progress_bar.progress(step / recursion_limit)
#                 if step >= recursion_limit:
#                     break
#         response = agent_state_value['response'] if agent_state_value else "No response found."
#     except langgraph.pregel.GraphRecursionError:
#         response = "The answer wasn't found in the data."
#     final_state = agent_state_value

#     st.write(f'The final answer is: {response}')
#     return response

# def main():
#     st.set_page_config(layout="wide")  # Use wide layout

#     st.title("Real-Time Agent Execution Visualization")
    
#     # Load your existing agent creation function
#     plan_and_execute_app = create_agent()

#     # Get the user's question
#     question = st.text_input("Enter your question:", "what did professor lupin teach?")

#     if st.button("Run Agent"):
#         inputs = {"question": question}
        
#         # Create columns for the keys of interest
#         col1, col2, col3, col4 = st.columns([1, 2, 2, 4])
        
#         with col1:
#             st.markdown("**Graph**")
#         with col2:
#             st.markdown("**plan**")
#         with col3:
#             st.markdown("**past_steps**")
#         with col4:
#             st.markdown("**aggregated_context**")

#         # Initialize placeholders for each column
#         placeholders = {
#             "plan": col2.empty(),
#             "past_steps": col3.empty(),
#             "aggregated_context": col4.empty(),
#         }
        
#         # Placeholder for the graph (use col1 for graph)
#         graph_placeholder = col1.empty()

#         response = execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder)
#         st.write("Final Answer:")
#         st.write(response)

# if __name__ == "__main__":
#     main()



import streamlit as st
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components



def create_network_graph(current_state):
    net = Network(directed=True, notebook=True)
    
    nodes = [
        {"id": "anonymize_question", "label": "anonymize_question"},
        {"id": "planner", "label": "planner"},
        {"id": "de_anonymize_plan", "label": "de_anonymize_plan"},
        {"id": "break_down_plan", "label": "break_down_plan"},
        {"id": "task_handler", "label": "task_handler"},
        {"id": "retrieve", "label": "retrieve"},
        {"id": "answer", "label": "answer"},
        {"id": "replan", "label": "replan"},
        {"id": "can_be_answered_already", "label": "can_be_answered_already"},
        {"id": "get_final_answer", "label": "get_final_answer"}
    ]
    
    edges = [
        ("anonymize_question", "planner"),
        ("planner", "de_anonymize_plan"),
        ("de_anonymize_plan", "break_down_plan"),
        ("break_down_plan", "task_handler"),
        ("task_handler", "retrieve"),
        ("task_handler", "answer"),
        ("retrieve", "replan"),
        ("answer", "replan"),
        ("replan", "can_be_answered_already"),
        ("can_be_answered_already", "get_final_answer")
    ]
    
    for node in nodes:
        color = "green" if node["id"] == current_state else "pink"
        net.add_node(node["id"], label=node["label"], color=color)
    
    for edge in edges:
        net.add_edge(*edge)
    
    return net

def save_and_display_graph(net):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name, notebook=True)
        tmp_file.flush()
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            return f.read()

def execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder, recursion_limit=15):
    config = {"recursion_limit": recursion_limit}
    agent_state_value = None
    progress_bar = st.progress(0)
    step = 0

    try:    
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            step += 1
            for _, agent_state_value in plan_output.items():
                # Update placeholders
                for key, placeholder in placeholders.items():
                    if key in agent_state_value and key != "curr_state":
                        if key in ["plan", "past_steps"] and isinstance(agent_state_value[key], list):
                            formatted_value = "\n".join([f"{i+1}. {item}" for i, item in enumerate(agent_state_value[key])])
                        else:
                            formatted_value = agent_state_value[key]
                        placeholder.markdown(f"{formatted_value}")
                
                current_state = agent_state_value["curr_state"]
                net = create_network_graph(current_state)
                graph_html = save_and_display_graph(net)
                
                # **Clear the placeholder before updating**
                graph_placeholder.empty()
                
                # **Update the graph using components.html **
                components.html(graph_html, height=600)  

                progress_bar.progress(step / recursion_limit)
                if step >= recursion_limit:
                    break
        response = agent_state_value['response'] if agent_state_value else "No response found."
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    final_state = agent_state_value

    st.write(f'The final answer is: {response}')
    return response

def main():
    st.set_page_config(layout="wide")  # Use wide layout

    st.title("Real-Time Agent Execution Visualization")
    
    # Load your existing agent creation function
    plan_and_execute_app = create_agent()

    # Get the user's question
    question = st.text_input("Enter your question:", "what did professor lupin teach?")

    if st.button("Run Agent"):
        inputs = {"question": question}
        
        # Create columns for the keys of interest
        col1, col2, col3, col4 = st.columns([1, 2, 2, 4])
        
        with col1:
            st.markdown("**Graph**")
        with col2:
            st.markdown("**plan**")
        with col3:
            st.markdown("**past_steps**")
        with col4:
            st.markdown("**aggregated_context**")

        # Initialize placeholders for each column
        placeholders = {
            "plan": col2.empty(),
            "past_steps": col3.empty(),
            "aggregated_context": col4.empty(),
        }
        
        # Placeholder for the graph (use col1 for graph)
        graph_placeholder = col1.empty()

        response = execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder)
        st.write("Final Answer:")
        st.write(response)

if __name__ == "__main__":
    main()