import fitz  # PyMuPDF
import requests
import weave
import os

# Initialize Weave for logging
weave.init('summarizer')


ENDPOINT_URL = "https://Mistral-small-achqr.eastus2.models.ai.azure.com/chat/completions"
PRIMARY_KEY = "your key"
DEFAULT_PAPER_URL = "https://arxiv.org/pdf/2407.20183.pdf"
IN_CONTEXT_FILE = "in_context_example.txt"

summary_length = 400  # Set the desired maximum length of the summary in words
chunk_size = 800  # Example chunk size in words
summary_pct = 10   # Example summary percentage for chunked summarization

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

def chunk_text_by_words(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def calculate_summary_length(chunk_text, summary_pct):
    word_count = len(chunk_text.split())
    return max(1, int(word_count * summary_pct / 100))

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

    # Chunked Summarization
    chunks = chunk_text_by_words(pdf_text, chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")

        # Calculate the dynamic summary length based on the chunk size
        summary_len = calculate_summary_length(chunk, summary_pct)

        prompt = (
            f"Summarize the following section of a research paper in {summary_len} words):\n{chunk}"
        )
        response = get_model_prediction(prompt)
        chunk_summaries.append(response.get("choices")[0].get("message").get("content"))

    # Combine chunk summaries and generate dynamic questions
    combined_summary = " ".join(chunk_summaries)
    prompt_for_questions = (
        f"In-context example:\n{in_context_example}\n\n"
        f"Generate a few key questions a researcher might ask about the following summarized sections:\n{combined_summary}"
    )
    questions_response = get_model_prediction(prompt_for_questions)

    if questions_response:
        generated_questions = questions_response.get("choices")[0].get("message").get("content")
        print("Generated Questions:")
        print(generated_questions)

        # Use the combined summary and generated questions to create the final prompt
        final_prompt = (
            f"In-context example:\n{in_context_example}\n\n"
            f"Please summarize the following text and address these questions :\n{FIXED_QUESTIONS}\n{generated_questions} (Word Limit: {summary_length} words)\n\n"
            f"Text:\n{combined_summary}"
        )

        # Get the final model prediction
        final_response = get_model_prediction(final_prompt)
        print("Model response:", final_response)
