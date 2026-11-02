
import pandas as pd
import os
import time

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# ----------------------------------
# 1. FILE PATHS
# ----------------------------------

PROMPT_FILE = "dataset/prompts.csv"
QUERY_FILE = "dataset/user_queries.csv"

RESPONSE_FILE = "dataset/chatbot_responses.csv"
RESULT_FILE = "dataset/chain_results.csv"

# ----------------------------------
# 2. CHECK DATASET FILES
# ----------------------------------

if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"Missing file: {PROMPT_FILE}")

if not os.path.exists(QUERY_FILE):
    raise FileNotFoundError(f"Missing file: {QUERY_FILE}")

prompts_df = pd.read_csv(PROMPT_FILE)
queries_df = pd.read_csv(QUERY_FILE)

print("Prompt templates loaded:", len(prompts_df))
print("User questions loaded:", len(queries_df))

# ----------------------------------
# 3. CONNECT TO OLLAMA MODEL
# ----------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3
)

# ----------------------------------
# 4. CHECK EXISTING RESULTS
# ----------------------------------


# Check existing result files safely

if os.path.exists(RESULT_FILE) and os.path.getsize(RESULT_FILE) > 0:
    results = pd.read_csv(RESULT_FILE).to_dict("records")
else:
    results = []

if os.path.exists(RESPONSE_FILE) and os.path.getsize(RESPONSE_FILE) > 0:
    responses = pd.read_csv(RESPONSE_FILE).to_dict("records")
else:
    responses = []


# Identify questions already completed
completed_ids = {
    str(row["question_id"])
    for row in results
    if row.get("status") == "Success"
}

# ----------------------------------
# 5. CHATBOT EXECUTION
# ----------------------------------

print("\n===== LANGCHAIN CHATBOT STARTED =====")

for index, query_row in queries_df.iterrows():

    question_id = str(
        query_row.get("question_id", index + 1)
    )

    question = str(
        query_row.get("question", query_row.get("query", ""))
    )

    template_name = str(
        query_row.get("template_name", "General Question")
    )

    if question_id in completed_ids:
        print(f"\nQuestion {question_id} already completed. Skipping.")
        continue

    matching_prompt = prompts_df[
        prompts_df["template_name"] == template_name
    ]

    if matching_prompt.empty:
        matching_prompt = prompts_df.iloc[[0]]

    template = matching_prompt.iloc[0]["template"]

    print(f"\nQuestion {question_id}: {question}")

    try:
        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm

        response = chain.invoke({"question": question})

        answer = response.content

        status = "Success"

        print("Answer:", answer)

    except Exception as error:

        answer = str(error)
        status = "Failed"

        print("Error:", error)

    # ----------------------------------
    # 6. SAVE CHATBOT RESPONSE
    # ----------------------------------

    responses.append({
        "question_id": question_id,
        "template_name": template_name,
        "question": question,
        "response": answer
    })

    pd.DataFrame(responses).to_csv(
        RESPONSE_FILE,
        index=False
    )

    # ----------------------------------
    # 7. SAVE CHAIN EXECUTION RESULT
    # ----------------------------------

    results.append({
        "question_id": question_id,
        "template_name": template_name,
        "question": question,
        "response": answer,
        "status": status
    })

    pd.DataFrame(results).to_csv(
        RESULT_FILE,
        index=False
    )

    print("Status:", status)
    print("Result saved successfully!")

# ----------------------------------
# 8. FINAL PERFORMANCE REPORT
# ----------------------------------

results_df = pd.DataFrame(results)

successful = (
    results_df["status"] == "Success"
).sum()

failed = (
    results_df["status"] == "Failed"
).sum()

print("\n===== TESTING COMPLETED =====")
print("Total questions:", len(queries_df))
print("Successful:", successful)
print("Failed:", failed)

print("\nChatbot responses saved to:", RESPONSE_FILE)
print("Chain results saved to:", RESULT_FILE)
