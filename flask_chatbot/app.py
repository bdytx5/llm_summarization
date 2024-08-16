
import requests
import json
import random
import os
from flask import Flask, request, jsonify, render_template
import weave

# Initialize Weave
client = weave.init('intro-example')

app = Flask(__name__)

# Define global variables for the models and cache
MODELS = [
    {"endpoint": "https://Meta-Llama-3-1-405B-Instruct-lqf.eastus2.models.ai.azure.com/chat/completions", "key": ""},
    {"endpoint": "https://Meta-Llama-3-1-70B-Instruct-xwib.eastus2.models.ai.azure.com/chat/completions", "key": ""},
    # {"endpoint": "your fourth model endpoint url", "key": "your fourth model key"},
    # {"endpoint": "your fifth model endpoint url", "key": "your fifth model key"},
]

IN_CONTEXT_FILE = "./in_context_example.txt"
SUMMARY_LENGTH = 400  # Maximum length of the summary in words
FIXED_QUESTIONS = """
What is the primary objective of this research?
What methodologies or algorithms are proposed or evaluated?
What datasets or experimental setups are used in this study?
What are the key findings and contributions of this research?
What are the implications of these findings for the broader field of AI?
What limitations or challenges are acknowledged by the authors?
What are the proposed future directions or next steps in this research?
"""

# Cache to store used prompts and models
used_prompts_cache = {}

@weave.op()
def get_model_prediction(prompt, endpoint, key, original_input=None):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    # Get the current call ID
    current_call = weave.get_current_call()
    call_id = current_call.id

    try:
        return parse_response(response.json()), call_id
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        return None, call_id

def parse_response(response):
    if 'choices' in response:
        choices = response['choices']
        for choice in choices:
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
                print(f"Model response content: {content}")
                return content
            if 'finish_reason' in choice:
                finish_reason = choice['finish_reason']
                print(f"Finish reason: {finish_reason}")
    return "No valid response"

def load_in_context_example(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return file.read()
    return ""

@weave.op()
def perform_self_reflection(summary, original_text, endpoint, key, original_input=None):
    reflection_prompt = (
        f"The following is a summary of the original document.  "
        f"Original Document:\n{original_text}\n\n"
        f"Summary:\n{summary}\n\n"
        f"If there are mistakes, simply remove incorrect sections and REWRITE it COMPLETELY as it was originally with the only change being the removal of incorrect sentences:"
        f"If everything is correct, simply rewrite the summary EXACTLY as it was:"
    )

    revised_summary, _ = get_model_prediction(reflection_prompt, endpoint, key)
    return revised_summary

def select_random_model(prompt):
    available_models = [model for model in MODELS if prompt not in used_prompts_cache.get(model['endpoint'], [])]

    if not available_models:
        # If all models have been used, reset the cache for the prompt
        for model in MODELS:
            if model['endpoint'] in used_prompts_cache:
                used_prompts_cache[model['endpoint']].remove(prompt)
        available_models = MODELS

    selected_model = random.choice(available_models)

    # Cache the selected model for the prompt
    if selected_model['endpoint'] not in used_prompts_cache:
        used_prompts_cache[selected_model['endpoint']] = []
    used_prompts_cache[selected_model['endpoint']].append(prompt)

    return selected_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']

    # Load the in-context example
    in_context_example = load_in_context_example(IN_CONTEXT_FILE)

    # Select a random model that hasn't been used with this prompt yet
    model = select_random_model(prompt)

    # Generate dynamic questions based on the input text
    prompt_for_questions = f"Generate key questions that a researcher would ask based on the following text:\n{prompt}"
    questions_response, _ = get_model_prediction(prompt_for_questions, model['endpoint'], model['key'])

    generated_questions = ""
    if questions_response:
        generated_questions = questions_response
    # Combine fixed and generated questions to create the final prompt
    final_prompt = (
        f"Heres a previous In-context example paper summary:\n{in_context_example}\n\n"
        f"Please summarize the following text and address these questions:\n{FIXED_QUESTIONS}\n{generated_questions} in {SUMMARY_LENGTH} words.\n\n"
        f"Text:\n{prompt}"
    )

    # Get the initial summary and call ID from the selected model
    summary, call_id = get_model_prediction(final_prompt, model['endpoint'], model['key'], original_input=prompt)

    # Perform self-reflection to remove factual errors
    revised_summary = perform_self_reflection(summary, prompt, model['endpoint'], model['key'], original_input=prompt)
    # revised_summary = summary
    return jsonify({
        "response": revised_summary, 
        "call_id": call_id, 
        "model_used": model['endpoint']
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    call_id = data['call_id']
    feedback_type = data['feedback']

    if feedback_type == "upvote":
        client.call(call_id).feedback.add_reaction("👍")
    elif feedback_type == "downvote":
        client.call(call_id).feedback.add_reaction("👎")

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
