import spacy
import numpy as np

# Charger les fichiers et les concaténer (Question 1)
with open("kaamelott_01.txt", encoding="utf-8") as f1, open("kaamelott_02.txt", encoding="utf-8") as f2:
    text1 = f1.readlines()
    text2 = f2.readlines()

text_merged = text1 + text2

# Charger le modèle spaCy et parser les lignes (Question 3)
nlp = spacy.load("fr_core_news_sm")
docs = [nlp(line.strip()) for line in text_merged]

lemmes = []
# Récupérer les lemmes et en faire un set (Question 4)
for doc in docs:
    for token in doc:
        lemmes.append(token.lemma_) 
lemmes = set(lemmes)

print(f"Nombre de lemmes uniques : {len(lemmes)}")

#Q5
num_lemmes = 7447
num_documents = 12711
taille_matrice = num_lemmes * num_documents
print(f"taille de la matrice terme_document : {taille_matrice}")

# Construire la matrice d'occurrence (Question 6)
lemma_to_index = {lemma: idx for idx, lemma in enumerate(lemmes)}
occurrence_matrix = np.zeros((len(lemmes), len(docs)), dtype=int)

for doc_idx, doc in enumerate(docs):
    for token in doc:
        if not token.is_punct and not token.is_stop:
            lemma_idx = lemma_to_index[token.lemma_]
            occurrence_matrix[lemma_idx, doc_idx] += 1


# print("Matrice d'occurrence :\n", occurrence_matrix)

#matrice TF
tf_matrix = np.zeros((len(lemmes), len(docs)), dtype=float)

for doc_idx, doc in enumerate(docs):
    total_tokens = len([token for token in doc if not token.is_punct and not token.is_stop])
    for token in doc:
        if not token.is_punct and not token.is_stop:
            lemma_idx = lemma_to_index[token.lemma_]
            tf_matrix[lemma_idx, doc_idx] = occurrence_matrix[lemma_idx, doc_idx] / total_tokens


print("Matrice TF :\n", tf_matrix)


##########################################################
# EXERCICE 3
##########################################################

# 1. 
df_t = np.sum(occurrence_matrix > 0, axis=1)

# Combien de documents contiennent le token « je » et « proverbe »
token_je = "je"
token_proverbe = "proverbe"
je_idx = lemma_to_index.get(token_je, -1)
proverbe_idx = lemma_to_index.get(token_proverbe, -1)

docs_with_je = df_t[je_idx] if je_idx != -1 else 0
docs_with_proverbe = df_t[proverbe_idx] if proverbe_idx != -1 else 0

print(f"Nombre de documents contenant 'je' : {docs_with_je}")
print(f"Nombre de documents contenant 'proverbe' : {docs_with_proverbe}")

# 2.
num_documents = len(docs)
normalized_df_t = df_t / num_documents

# 3.
idf_t = np.log(num_documents / (df_t)) 

idf_dict = {lemma: idf_t[idx] for lemma, idx in lemma_to_index.items()}

# Afficher un aperçu de quelques scores IDF
print("Aperçu des scores IDF :")
for lemma, idf_value in list(idf_dict.items())[:10]:
    print(f"{lemma}: {idf_value}")


##########################################################
# EXERCICE 4
##########################################################

 