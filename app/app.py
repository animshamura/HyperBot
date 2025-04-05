#!pip install transformers gradio torch --quiet for colab

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import gradio as gr
import torch

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Chat function
def vanilla_chatbot(message, history):
    inputs = tokenizer([message], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return response

# Gradio Chat Interface
demo_chatbot = gr.ChatInterface(
    vanilla_chatbot,
    title="HyperBot",
    description="Enter text to start chatting with HyperBot!",
)

# Launch the app
demo_chatbot.launch(share=True)  # Use share=True to get a public link
