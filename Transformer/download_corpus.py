from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

with open("my_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dataset['train']['text']))

print("my_corpus.txt created successfully!")
