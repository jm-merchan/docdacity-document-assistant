from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT
from langgraph.checkpoint.memory import InMemorySaver


# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph
# workflow.  See README.md Task 2.1 for detailed explanations of
# each property.
class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    # Annotated messages list that adds messages to the state
    # Annotated takes care of type hinting and adding messages to the state
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # TODO: Modify actions_taken to use an operator.add reducer
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used


# TODO: Implement the classify_intent function.
# This function should classify the user's intent and set the next step in the workflow.
# Refer to README.md Task 2.2
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # TODO Configure the llm chat model for structured output
    # Use llm.with_structured_output(UserIntent) for structured responses
    structured_llm = llm.with_structured_output(UserIntent)

    # TODO Create a formatted prompt with conversation history and user input
    # invoke the prompt template providing as input the user input and the chat history
    """
    Create a prompt by calling the get_intent_classification_prompt() function from prompts.py.(HINT: you will need to call format on the returned value and pass in the user_input and conversation_history)
    """
    prompt_with_history = get_intent_classification_prompt().format(
        # format the prompt with the user input and conversation history
        user_input=state["user_input"],
        conversation_history=state.get("messages", []),
    )

    response = structured_llm.invoke(prompt_with_history)
    # Update the state with actions_taken = ["classify_intent"] also include the new intent value and next_step, then return the updated state
    # The function should return a state update with actions_taken, intent, and next_step
    return {
        "actions_taken": ["classify_intent"],
        "intent": response,
        "next_step": "qa" if response.intent_type == "qa" else "summarization" if response.intent_type == "summarization" else "calculation" if response.intent_type == "calculation" else "qa",
    }

def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    # Get the LLM from the config
    # the llm is the first item in the configurable dictionary
    llm = config.get("configurable").get("llm")
    # Get the tools from the config
    tools = config.get("configurable").get("tools")

    # Get the prompt template for the qa agent
    prompt_template = get_chat_prompt_template("qa")
    # Invoke the prompt template with the user input and conversation history
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)
    # Update the state with actions_taken = ["qa_agent"], current_response = result, tools_used = tools_used, and next_step = "update_memory"
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the summarization_agent function. Refer to README.md Task 2.3
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    # Get the LLM from the config
    # the llm is the first item in the configurable dictionary
    llm = config.get("configurable").get("llm")
    # Get the tools from the config
    tools = config.get("configurable").get("tools")

    # Get the prompt template for the summarization agent
    prompt_template = get_chat_prompt_template("summarization")
    # Invoke the prompt template with the user input and conversation history
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }

# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    # Get the LLM from the config
    # the llm is the first item in the configurable dictionary
    llm = config.get("configurable").get("llm")
    # Get the tools from the config
    tools = config.get("configurable").get("tools")

    # Get the prompt template for the calculation agent
    prompt_template = get_chat_prompt_template("calculation")
    # Invoke the prompt template with the user input and conversation history
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }

# TODO: Finish implementing the update_memory function. Refer to README.md Task 2.4
def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    # TODO: Retrieve the LLM from config
    # the llm is the first item in the configurable dictionary
    llm = config.get("configurable").get("llm")
    # Get the tools from the config
    tools = config.get("configurable").get("tools")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)
    # Pass in the correct schema from scheams.py to extract conversation summary, active documents

    response = structured_llm.invoke(prompt_with_history)
    return {
        "conversation_summary":  response.summary, # TODO: Extract summary from response
        "active_documents":  response.document_ids, # TODO: Update with the current active documents
        "next_step": "end", # TODO: Update the next step to end
        "actions_taken": ["update_memory"],
    }

def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")

# TODO: Complete the create_workflow function. Refer to README.md Task 2.5
def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    # TODO: Add all the nodes to the workflow by calling workflow.add_node(...)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    # Graph structure classify_intent --> [qa_agent|summarization_agent|calculation_agent] --> update_memory --> END

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            # TODO: Map the intent strings to the correct node names
            "qa": "qa_agent",
            "summarization": "summarization_agent",
            "calculation": "calculation_agent",
            "end": END
        }
    )

    # TODO: For each node add an edge that connects it to the update_memory node
    # qa_agent -> update_memory
    workflow.add_edge("qa_agent", "update_memory") # qa_agent -> update_memory
    # summarization_agent -> update_memory
    workflow.add_edge("summarization_agent", "update_memory") # summarization_agent -> update_memory
    # calculation_agent -> update_memory
    workflow.add_edge("calculation_agent", "update_memory") # calculation_agent -> update_memory

    workflow.add_edge("update_memory", END)


    return workflow.compile(checkpointer=InMemorySaver())