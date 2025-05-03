import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from utils import PoSStripper

if __name__ == '__main__':
    only_essential_parts_of_speech = False

    use_full_verses = True

    # Loading from HuggingFace existing/saved models
    model = SentenceTransformer('all-mpnet-base-v2')

    if use_full_verses:
        df = pd.read_csv('asv.csv')
    elif only_essential_parts_of_speech:
        df = pd.read_csv('asv_chapters_stripped.csv')
    else:
        df = pd.read_csv('asv_chapters.csv')

    corpus = []
    for key, value in df.iterrows():
        corpus.append(value['Text'])

    corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True)

    query = "Did the Jews or the Romans kill Christ?"
    if only_essential_parts_of_speech:
        query = PoSStripper().strip(query)

    processed_query = re.sub(r'[^\w\s]', ' ', query)

    query_embedding = model.encode(processed_query, show_progress_bar=True, convert_to_tensor=True)

    result = util.semantic_search(query_embedding, corpus_embeddings)
    print(result)
