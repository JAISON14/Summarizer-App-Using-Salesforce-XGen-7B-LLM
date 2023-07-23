# XGen-7b Text Summarizer
This is a Python web application that utilizes the XGen-7b fully open-source language model (LLM) by Salesforce to summarize text provided by the user. The application is deployed as a web app on Microsoft Azure, making it easily accessible and user-friendly.

## Overview
XGen-7b is a powerful language model that can generate high-quality summaries of input text. In this web app, the user interacts with an artificial intelligence assistant through a user-friendly interface. The assistant takes the user's input text, provided as an article or any written content, and generates a concise and informative summary.

The summarization process is handled by the XGen-7b model, which has been fine-tuned for causal language modeling. The model is equipped with specific configurations, such as using torch's bfloat16 data type for enhanced performance.

### How to Use the App

1.Open the web app URL: [[Web App URL]](https://summarizerapp.azurewebsites.net/)

2.Upon accessing the app, you will see a text box labeled "Text."

3.Enter the text you want to summarize into the "Text" box. The input text can be a paragraph, article, or any written content you wish to summarize.

4.Click on the "Summarize" button to initiate the summarization process.

5.The XGen-7b assistant will process your input text and generate a helpful summary.

6.The generated summary will be displayed in the "Summary" text box below the input box.


## Important Note
The XGen-7b model's performance heavily depends on the input text and the context of the content provided. While the model strives to produce informative summaries, it may not always capture the nuances or context perfectly. Therefore, we recommend using the summary as a reference and always verify its accuracy, especially for critical tasks.

## About XGen-7b
XGen-7b is a fully open-source language model developed by Salesforce. It is designed to perform various natural language processing tasks, including text generation and summarization. The model has been fine-tuned on extensive data and is equipped with the latest advancements in language modeling.

## Deployed on Azure
This app is deployed on Microsoft Azure as a web app, making it easily accessible through any web browser. The deployment ensures high availability and scalability, allowing multiple users to interact with the summarization model concurrently.
