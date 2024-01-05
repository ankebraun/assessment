"""
Question and answering system about Llama-2 
Some part of the code was adapted from 
https://github.com/openai/openai-cookbook/blob/b7316a18eb4b24ab8b597039a56c37fd2cebcb90/examples/Question_answering_using_embeddings.ipynb
"""

import os
import openai
import urllib
import xml.etree.ElementTree as ET
import pandas as pd
import tiktoken # for converting embeddings saved as strings back to arrays# for calling the OpenAI API # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = os.environ["OPENAI_API_KEY"]


def fetch_papers():
    """Fetches papers from the arXiv API and returns them as a list of strings."""
    url = (
        "http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70"
    )
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    root = ET.fromstring(data)
    papers_list = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        paper_info = f"Title: {title}\nSummary: {summary}\n"
        papers_list.append(paper_info)
    return papers_list


papers = fetch_papers()

embeddings = []
for i in range(0, len(papers)):
    # create embeddings for each paper
    response = openai.embeddings.create(model="text-embedding-ada-002", input=papers[i])
    # extract embeddings and append to list of embeddings for all papers
    embeddings.append(response.data[0].embedding)

# create pandas dataframe with text and embeddings as columns
df = pd.DataFrame({"text": papers, "embedding": embeddings})


# search function that takes user query and the dataframe and uses distance between
# query embedding and text embeddings to rank the texts from the papers by relatedness
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses,
    sorted from most related to least."""
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the data to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nQuestions on Llama-2:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Llama-2."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


question = input("What is your question? ")
print(ask(question))
