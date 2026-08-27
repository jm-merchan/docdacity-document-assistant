# DocDacity Document Assistant

Project for *Agentic AI Engineer with LangChain and LangGraph*.

DocDacity is a small document assistant: you ask a question, the graph figures out if it is Q&A, a summary, or a calculation, and then the matching node runs with the document tools. The sample corpus is invoices, one contract, and a claim.

```
classify_intent → qa_agent | summarization_agent | calculation_agent → update_memory → END
```

## Run it

Code is in `doc_assistant_project/`. You need Python 3.9+ and an `OPENAI_API_KEY`.

```bash
cd doc_assistant_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env         # put the key in here
python main.py
```

The first successful turn creates `logs/` (tool calls) and `sessions/` (one JSON per session). In the REPL you can use `/docs`, `/help`, and `/quit`.

## What I implemented

### Schemas (`src/schemas.py`)

I used Pydantic so the LLM has to fill a real object instead of a blob of text.

`UserIntent` is the important one for routing: `intent_type` can only be `qa`, `summarization`, `calculation`, or `unknown`. Confidence is a float between 0 and 1 (`ge=0.0, le=1.0`). There is also a `reasoning` field so you can see why it picked that label.

`AnswerResponse` is question, answer, sources, confidence, and a timestamp that defaults to now. Summarization, calculation, and the memory update have their own models along the same lines.

### Graph (`src/agent.py`)

`create_workflow` wires the five nodes and compiles with `InMemorySaver`. Without the checkpointer the next message would not see the previous graph state (state gets reset).

`classify_intent` calls `llm.with_structured_output(UserIntent)` and writes `next_step`. I map `qa` / `summarization` / `calculation` onto the three agent nodes. If the label is `unknown` I send it to Q&A (that is what the assignment default asked for), even though the prompt still has an `unknown` category.

Each agent node loads its prompt from `get_chat_prompt_template`, runs the ReAct agent with tools, and returns `actions_taken`. That field uses `operator.add`, so you get the full path for the turn (`classify_intent`, then the agent, then `update_memory`) instead of only the last node.

`llm` and `tools` are not stored in state. They come in through `config["configurable"]` in `process_message`, together with `thread_id` = session id. 
> State is the conversation; config is how this thread is executed.

### Calculator (`src/tools.py`)

`calculator` is a `@tool`. Before `eval` I reject anything that is not digits and `+ - * / ( ) . %`. The return value is always a string. Bad input or a failed eval gets logged and comes back as an error message, not an exception to the graph.

The calculation prompt tells the model to use the tool even for easy arithmetic. In practice it did that on the invoice-sum query.

### Prompts (`src/prompts.py`)

The classifier prompt lists the four categories and asks for confidence plus a short reason. `get_chat_prompt_template(intent_type)` swaps the system prompt: QA, summarization, or calculation. Same chat shape in all three cases (system, history, user input).

One thing I noticed while testing: "What's the total amount in invoice INV-001?" went to `calculation`, not `qa`, because the classifier prompt treats number questions as calculation. That is consistent with the prompt, just not with the example list in the assignment README. For a clean `qa` turn I asked who the client was instead.

### Memory

Two stores, because they do different jobs:

- `InMemorySaver` keeps LangGraph state for the `thread_id` (messages, summary, documents).
- `sessions/<id>.json` is the `SessionState` file so you can come back after restarting the process.

`update_memory` writes the summary and active document ids into the graph. `_save_session` writes the file. Tool history is separated, under `logs/`.

## How I tested

```bash
cd doc_assistant_project
python main.py
```

User id `jose`. I ran the three assignment examples plus a fourth question after the first one classified as calculation. After a turn you should see `INTENT`, `TOOLS USED` when tools ran, and a `CONVERSATION SUMMARY`. `logs/` and `sessions/` should have new files.

I also checked the calculator locally: `calc.invoke("2 + 3")` returns a string, and `__import__('os')` is rejected.

## Example runs

Output copied from `python main.py`. First block is session `f6c7aa3b-5e1f-4e6d-a540-cc58d4efad69`. The Q&A turn is a second run (`0c183da3-9a1b-4be9-b3a2-c8e78886023a`) after the invoice-total question classified as calculation.

```
Enter Message: What's the total amount in invoice INV-001?

Processing...

🤖 Assistant: The total amount in invoice INV-001 is $22,000, which includes a subtotal of $20,000 and a tax of $2,000.

INTENT: calculation

TOOLS USED: document_reader

CONVERSATION SUMMARY: The user inquired about the total amount in invoice INV-001. The invoice details include services rendered such as consulting services ($5,000), software development ($12,500), and support & maintenance ($2,500), leading to a subtotal of $20,000. A tax of 10% ($2,000) is applied, resulting in a total amount of $22,000.

Enter Message: Summarize all contracts

Processing...
Deserializing unregistered type schemas.UserIntent from checkpoint. This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true to block now, or add to allowed_msgpack_modules to allow explicitly: [('schemas', 'UserIntent')]
Deserializing unregistered type schemas.CalculationResponse from checkpoint. This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true to block now, or add to allowed_msgpack_modules to allow explicitly: [('schemas', 'CalculationResponse')]

🤖 Assistant: ### Summary of Contract Document (ID: CON-001)

**Title:** Service Agreement  
**Date:** January 1, 2024  
**Parties Involved:**  
- **Provider:** DocDacity Solutions Inc.  
- **Client:** Healthcare Partners LLC  

**Services Provided:**  
1. Document Processing Platform Access  
2. 24/7 Technical Support  
3. Monthly Data Analytics Reports  
4. Compliance Monitoring  

**Financial Details:**  
- **Monthly Fee:** $15,000  
- **Total Contract Value:** $180,000  

**Duration:** 12 months  

**Termination Clause:** Either party may terminate the agreement with 60 days written notice.  

This contract outlines the services and financial terms agreed upon between DocDacity Solutions Inc. and Healthcare Partners LLC for a period of one year.

INTENT: summarization

TOOLS USED: document_reader, document_search, document_reader

CONVERSATION SUMMARY: The conversation involved retrieving and summarizing a service agreement contract (Document ID: CON-001) between DocDacity Solutions Inc. and Healthcare Partners LLC. The contract, dated January 1, 2024, includes services such as document processing platform access, 24/7 technical support, monthly data analytics reports, and compliance monitoring. The contract is valued at $180,000 over 12 months, with a monthly fee of $15,000. It includes a termination clause allowing either party to terminate with 60 days' notice.

Enter Message: Calculate the sum of all invoice totals

Processing...
Deserializing unregistered type schemas.SummarizationResponse from checkpoint. This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true to block now, or add to allowed_msgpack_modules to allow explicitly: [('schemas', 'SummarizationResponse')]

🤖 Assistant: The sum of all invoice totals is $305,800.

INTENT: calculation

TOOLS USED: document_reader, document_search, document_reader, document_search, document_reader, document_reader, calculator

CONVERSATION SUMMARY: The sum of all invoice totals is $305,800, calculated from the following invoices:

1. **Invoice #12345 (ID: INV-001):** Total $22,000
2. **Invoice #12346 (ID: INV-002):** Total $69,300
3. **Invoice #12347 (ID: INV-003):** Total $214,500

Enter Message: /docs

AVAILABLE DOCUMENTS:
----------------------------------------
ID: INV-001
Title: Invoice #12345
Type: invoice
----------------------------------------
ID: CON-001
Title: Service Agreement
Type: contract
Value: $180,000.00
----------------------------------------
ID: CLM-001
Title: Insurance Claim #78901
Type: claim
Amount: $2,450.00
----------------------------------------
ID: INV-002
Title: Invoice #12346
Type: invoice
Total: $69,300.00
----------------------------------------
ID: INV-003
Title: Invoice #12347
Type: invoice
Total: $214,500.00
----------------------------------------
```

Q&A session:

```
Enter Message: Who is the client listed on invoice INV-001?

Processing...

🤖 Assistant: The client listed on invoice INV-001 is Acme Corporation.

INTENT: qa

TOOLS USED: document_reader

CONVERSATION SUMMARY: The client listed on invoice INV-001 is Acme Corporation.
```

## Layout

```
Report-Building Agent/
├── README.md
└── doc_assistant_project/
    ├── main.py
    ├── requirements.txt
    ├── src/          # schemas, agent, assistant, prompts, tools, retrieval
    ├── logs/         # created at runtime
    └── sessions/     # created at runtime
```

