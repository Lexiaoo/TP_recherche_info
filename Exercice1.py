import spacy

nlp = spacy.load("fr_core_news_sm")

doc = nlp("Glace au chocolat.")

for token in doc:
    print(f"{token.text}: {token.pos_}")
    print(f"{token.lemma_}")

print(type (doc[0]))