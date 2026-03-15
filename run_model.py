import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------
# Load Cybersecurity Keywords
# ---------------------------------
with open("cyber_keywords.json", "r") as f:
    cyber_data = json.load(f)

keywords = []
for category in cyber_data["cybersecurity_keywords"].values():
    keywords.extend(category)

keywords = set(k.lower() for k in keywords)

# ---------------------------------
# Greetings
# ---------------------------------
greetings = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "good afternoon"
]

# ---------------------------------
# Classify Query
# ---------------------------------
def classify_query(query):

    query = query.lower()

    for g in greetings:
        if g in query:
            return "greeting"

    for keyword in keywords:
        if keyword in query:
            return "cyber"

    return "blocked"


# ---------------------------------
# Load Model
# ---------------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model.eval()

print("Model loaded successfully")


# ---------------------------------
# System Prompt
# ---------------------------------
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
politely refuse to answer.
"""


# ---------------------------------
# Ask Model
# ---------------------------------
def ask_model(question):

    query_type = classify_query(question)

    if query_type == "greeting":
        return "Hello 👋 I am a cybersecurity assistant. Ask me anything about cybersecurity."

    if query_type == "blocked":
        return "❌ This assistant only answers cybersecurity questions."

    prompt = SYSTEM_PROMPT + "\n\nUser: " + question + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n🤖 Thinking...\n")


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    response = response.replace(prompt, "").strip()

    return response


# ---------------------------------
# Chat Loop
# ---------------------------------
if __name__ == "__main__":

    print("\nCybersecurity AI Assistant Ready")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("Ask: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        answer = ask_model(user_input)

        print("\nResponse:\n", answer)
