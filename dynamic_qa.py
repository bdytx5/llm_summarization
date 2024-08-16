import fitz  # PyMuPDF
import requests
import os

ENDPOINT_URL = "https://Mistral-small-achqr.eastus2.models.ai.azure.com/chat/completions"
PRIMARY_KEY = "eYdHRarS5uMpS8iXfSB3dqEzi9gcUuui"
DEFAULT_PAPER_URL = "https://arxiv.org/pdf/2407.20183.pdf"
IN_CONTEXT_FILE = "in_context_example.txt"

summary_length = 400  # Set the desired maximum length of the summary in words

FIXED_QUESTIONS = """
What is the primary objective of this research?
What methodologies or algorithms are proposed or evaluated?
What datasets or experimental setups are used in this study?
What are the key findings and contributions of this research?
What are the implications of these findings for the broader field of AI?
What limitations or challenges are acknowledged by the authors?
What are the proposed future directions or next steps in this research?
"""

@weave.op()
def get_model_prediction(prompt):
    headers = {
        "Authorization": f"Bearer {PRIMARY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    return response.json()

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded PDF from {url}")

def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def load_in_context_example(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return file.read()
    return ""

if __name__ == "__main__":
    # Download the PDF if it doesn't exist
    pdf_path = "2407.20183.pdf"
    if not os.path.exists(pdf_path):
        download_pdf(DEFAULT_PAPER_URL, pdf_path)

    # Load the PDF text
    pdf_text = load_pdf_text(pdf_path)
    print("PDF text loaded.")

    # Load the in-context example
    in_context_example = load_in_context_example(IN_CONTEXT_FILE)

    # Generate dynamic questions
    prompt_for_questions = f"Generate key questions that a researcher would ask based on the following text:\n{pdf_text}"
    questions_response = get_model_prediction(prompt_for_questions)

    if questions_response:
        generated_questions = questions_response.get("choices")[0].get("message").get("content")
        print("Generated Questions:")
        print(generated_questions)

        # Combine fixed and generated questions to create the final prompt
        final_prompt = (
            f"Heres a previous In-context example paper summary:\n{in_context_example}\n\n"
            f"Please summarize the following text and address these questions:\n{FIXED_QUESTIONS}\n{generated_questions} in {summary_length} words \n\n"
            f"Text:\n{pdf_text}"
        )

        # Get the final model prediction
        final_response = get_model_prediction(final_prompt)
        print("Model response:", final_response)
