import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------
# Load Cybersecurity Keywords
# -----------------------------
with open("cyber_keywords.json", "r") as f:
    cyber_data = json.load(f)

# Flatten all categories into a single list
keywords = []
for category in cyber_data["cybersecurity_keywords"].values():
    keywords.extend(category)

# -----------------------------
# Function to check if query is cybersecurity related
# -----------------------------
def is_cybersecurity_query(query):
    query_lower = query.lower()
    for keyword in keywords:
        if keyword.lower() in query_lower:
            return True
    return False

# -----------------------------
# Load Mistral 7B Model
# -----------------------------
model_name = "mistralai/Mistral-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

# -----------------------------
# System Prompt (Optional but recommended)
# -----------------------------
SYSTEM_PROMPT = """
You are a cybersecurity expert assistant.
You only answer questions about:
- malware
- threat intelligence
- vulnerabilities
- digital forensics
- incident response
- network security
- cloud security

If the question is outside cybersecurity, refuse to answer.
"""

# -----------------------------
# Query the model
# -----------------------------
def ask_model(user_question, max_tokens=300):
    # Step 1: Check if query is cybersecurity related
    if not is_cybersecurity_query(user_question):
        return "This assistant only answers cybersecurity related questions."

    # Step 2: Combine system prompt + user question
    prompt = SYSTEM_PROMPT + "\n\nQuestion: " + user_question + "\nAnswer:"

    # Step 3: Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt part from the answer
    answer = answer.replace(prompt, "").strip()
    return answer

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = ask_model(user_input)
        print("\nResponse:", response)