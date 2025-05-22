import pandas as pd
from utils.create_chapter_csv import read_corpus
from hugging_face_training.chapter_one_basic_examples import translation


if __name__ == '__main__':
    # reading in the American Standard Version of the Bible separated as verses
    df = pd.read_csv('asv.csv')
    read_corpus(df)
    # for now, first available text
    input_text = str(df.values[0][5]).strip()
    print(input_text)
    print()
    translation(input_text)