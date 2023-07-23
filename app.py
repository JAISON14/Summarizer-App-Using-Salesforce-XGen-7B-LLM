# Import necessary libraries
import torch
import gradio as gr 
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained tokenizer and model from the Salesforce xgen-7b-8k-inst model
tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/xgen-7b-8k-inst",
    trust_remote_code=True,
)

# Load the pre-trained model for causal language modeling with specific configurations
model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-inst",
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
)

# Function to summarize the input text using the pre-trained model
def summarize(text):
    # Create a header for the conversation between human and AI assistant
    header = (
        "A chat between a curious human and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the human's questions. \n\n"
    )

    # Combine the header with the user-provided text and format it for model input
    text = header + "### Human: Please summarize the following article. \n\n" + text + "\n###"

    # Tokenize the input text and convert it to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt")

    # Generate a summary using the pre-trained model
    generated_ids = model.generate(
        **inputs,
        max_length=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    )
    # Decode the generated summary and remove special tokens
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True).lstrip()

    # summary starts from ### Assistant: and ends with <|endoftext|>
    summary = summary.split("### Assistant:")[1]
    summary = summary.split("<|endoftext|>")[0]

    # Return the summary as the output to be displayed in the Gradio interface
    return gr.Textbox.update(value=summary)

# Create a Gradio interface for the summarize function
with gr.BLocks() as demo:
    # Add a textbox to input the text to be summarized
    with gr.Row():
        text = gr.Textbox(lines=20, label="Text")
        # Add another textbox to display the generated summary
        summary = gr.Textbox(label="Summary", lines=20)
    # Add a button to trigger the summarization process
    submit = gr.Button(text="Summarize")
    # Link the button click event to the summarize function, with text as input and summary as output
    submit.click(summarize, inputs=text, outputs=summary)

# Launch the Gradio interface
demo.launch()
