import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from utils import PoSStripper

if __name__ == '__main__':
    only_essential_parts_of_speech = False

    # Loading from HuggingFace existing/saved models
    model = SentenceTransformer('all-mpnet-base-v2')

    if only_essential_parts_of_speech:
        df = pd.read_csv('asv_chapters_stripped.csv')
    else:
        df = pd.read_csv('asv_chapters.csv')

    corpus = []
    for key, value in df.iterrows():
        corpus.append(value['Text'])

    corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True)

    query = "Would the Lord allow women priests?"
    if only_essential_parts_of_speech:
        query = PoSStripper().strip(query)

    processed_query = re.sub(r'[^\w\s]', ' ', query)

    query_embedding = model.encode(processed_query, show_progress_bar=True, convert_to_tensor=True)

    result = util.semantic_search(query_embedding, corpus_embeddings)
    print(result)
