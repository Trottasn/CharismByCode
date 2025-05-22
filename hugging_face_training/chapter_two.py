from huggingface_hub import list_datasets

if __name__ == "__main__":
    # all datasets available with HuggingFace
    all_datasets = list(list_datasets())
    # print(f"There are {len(all_datasets)} datasets currently available.")
    print(f"The first 10 are: {all_datasets[:10]}")
