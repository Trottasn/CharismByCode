# Regex
import re
# Preprocessing
import pandas as pd
# PoS stripping for optional preprocessing
from utils.utils import PoSStripper


def read_corpus(df, only_essential_parts_of_speech=False):
    actual_chapter = 0
    total_chapter = 0
    book_name = "Genesis"
    full_chapter_text = ''
    continuous_output_data_frame = pd.DataFrame(columns=['Book Name', 'Chapter', 'Text'])
    stripper = PoSStripper()
    for key, value in df.iterrows():
        if not isinstance(value['Text'], str):
            continue
        if actual_chapter == value['Chapter']:
            full_chapter_text += value['Text']
        else:
            if only_essential_parts_of_speech:
                stripped_text = stripper.strip(full_chapter_text)
            else:
                stripped_text = full_chapter_text
            stripped_text = re.sub(r'[^\w\s]', ' ', stripped_text)
            new_dataframe = pd.DataFrame([[book_name, actual_chapter, stripped_text]], columns=['Book Name', 'Chapter', 'Text'])
            new_dataframe.index.values[0] = total_chapter
            continuous_output_data_frame = pd.concat([continuous_output_data_frame, new_dataframe], ignore_index=False)
            total_chapter += 1
            book_name = value['Book Name']
            actual_chapter = value['Chapter']
            full_chapter_text = value['Text']
    continuous_output_data_frame.index.rename('Chapter ID', inplace=True)
    if only_essential_parts_of_speech:
        continuous_output_data_frame.to_csv('asv_chapters_stripped.csv')
    else:
        continuous_output_data_frame.to_csv('asv_chapters.csv')
    return continuous_output_data_frame
