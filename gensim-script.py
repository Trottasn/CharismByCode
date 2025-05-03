import re
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

df = pd.read_csv('asv.csv')

def read_corpus():
    actual_chapter = 0
    total_chapter = 0
    full_chapter_text = ""
    for key, value in df.iterrows():
        if not isinstance(value['Text'], str):
            continue
        if actual_chapter == value['Chapter']:
            full_chapter_text += value['Text']
        print(full_chapter_text)
        stripped_text = re.sub(r'[^\w\s]', '', full_chapter_text)
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(stripped_text), [total_chapter])
        actual_chapter = value['Chapter']
        total_chapter += 1
        full_chapter_text = value['Text']

train_corpus = list(read_corpus())

def train_and_test(verse):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=200)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=200)

    models = [
        # PV-DBOW (Skip-Gram equivalent of Word2Vec)
        Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=100),

        # PV-DM w/average (CBOW equivalent of Word2Vec)
        Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, min_count=10, epochs=100),
    ]

    documents = train_corpus
    models[0].build_vocab(documents)
    models[1].reset_from(models[0])

    for model in models:
       model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    new_model = ConcatenatedDoc2Vec((models[0], models[1]))

    inferred_vector = model.infer_vector(train_corpus[verse].words)
    sims = model.docvecs.most_similar([inferred_vector])
    print(sims)
    # model 1
    inferred_vector = new_model.models[0].infer_vector(train_corpus[verse].words)
    sims2 = new_model.models[0].docvecs.most_similar([inferred_vector])
    print(sims2)
    # model 2
    inferred_vector = new_model.models[1].infer_vector(train_corpus[verse].words)
    sims3 = new_model.models[1].docvecs.most_similar([inferred_vector])
    print(sims3)

if __name__ == '__main__':
    # train_and_test(100)
    print(' '.join(train_corpus[100][0]) + '\n')
    print(' '.join(train_corpus[7559][0]) + '\n')
