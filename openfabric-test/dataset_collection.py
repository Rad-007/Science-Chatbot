from transformers import pipeline

import pinecone


#Load and Prepare Dataset


from datasets import load_dataset

from tqdm.auto import tqdm

import pandas as pd

# load the dataset from huggingface in streaming mode and shuffle it
wiki_data = load_dataset(
    'vblagoje/wikipedia_snippets_streamed',
    split='train',
    streaming=True
).shuffle(seed=960)



science = wiki_data.filter(
    lambda d: d['section_title'].startswith('Science')
)


total_doc_count=50000

counter = 0
docs = []
# iterate through the dataset and apply our filter
for d in tqdm(science, total=total_doc_count):
    # extract the fields we need
    doc = {
        "article_title": d["article_title"],
        "section_title": d["section_title"],
        "passage_text": d["passage_text"]
    }
    # add the dict containing fields we need to docs list
    docs.append(doc)

    # stop iteration once we reach 50k
    if counter == total_doc_count:
        break

    # increase the counter on every iteration
    counter += 1


df=pd.DataFrame(docs)

df.head()


import torch
from sentence_transformers import SentenceTransformer

pinecone.init(
    api_key='5de1f5b6-b6da-429f-a89d-3496e61d9432',
    enviroment='us-west1-gcp'
)


index_name="abstractive-question-answering"

if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric="cosine"
    )

# connect to abstractive-question-answering index we created
index = pinecone.Index(index_name)


#Initialize 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)
retriever


#Generate embedding



batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # extract batch
    batch = df.iloc[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch["passage_text"].tolist()).tolist()
    # get metadata
    meta = batch.to_dict(orient="records")
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()



from transformers import BartTokenizer, BartForConditionalGeneration

# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)
