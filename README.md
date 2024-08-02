# Chat with your documents

Chat anything about the documents that you provided. This is using open source LLM models by Meta: Llama 3

![ChatbotUI](/images/ChatbotUI2.png)

## Install dependencies

Install the required dependencies in the requirements.txt

## Get Groq API Key

Get your groq API Key. [Obtain API KEY here](https://console.groq.com/keys)

Set environment variable GROQ_API_KEY:

```
export GROQ_API_KEY="your_api_key"
```


## Upload files to the data directory

Go to the /data directory and upload your documents.

run the populate_database script:

```
python populate_database.py
```

## Run the streamlit app

```
streamlit run queryprompt.py
```
