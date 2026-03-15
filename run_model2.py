import json
import ollama

# -----------------------------
# Load Cybersecurity Keywords
# -----------------------------
with open("cyber_keywords.json", "r") as f:
    cyber_data = json.load(f)

keywords = []
for category in cyber_data["cybersecurity_keywords"].values():
    keywords.extend(category)

keywords = set(k.lower() for k in keywords)

# -----------------------------
# Greetings
# -----------------------------
greetings = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening"
]

# -----------------------------
# Query Classification
# -----------------------------
def classify_query(query):

    query = query.lower()

    for g in greetings:
        if g in query:
            return "greeting"

    for keyword in keywords:
        if keyword in query:
            return "cyber"

    return "blocked"


# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a cybersecurity expert assistant.

You ONLY answer questions about:
- malware
- vulnerabilities
- threat intelligence
- incident response
- digital forensics
- network security
- cloud security

If the question is unrelated to cybersecurity,
politely refuse.
"""


# -----------------------------
# Ask Model
# -----------------------------
def ask_model(question):

    query_type = classify_query(question)

    if query_type == "greeting":
        return "Hello 👋 I am a cybersecurity AI assistant."

    if query_type == "blocked":
        return "❌ This assistant only answers cybersecurity questions."

    print("\n🤖 Thinking...\n")

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )

    return response["message"]["content"]


# -----------------------------
# Chat Loop
# -----------------------------
if __name__ == "__main__":

    print("\nCybersecurity AI Assistant Ready")
    print("Type 'exit' to quit\n")

    while True:

        question = input("Ask: ")

        if question.lower() in ["exit","quit"]:
            break

        answer = ask_model(question)

        print("\nResponse:\n", answer)
