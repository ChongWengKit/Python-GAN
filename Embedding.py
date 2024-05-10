
import numpy as np

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pickle


file_path = 'D:/output_train_bird.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
texts = []
counter=0
for line in lines:
    parts = line.split('|')
    if len(parts) > 1:
        text = parts[1].strip()
        if text:
            texts.append(text)

tokenized_texts = [word_tokenize(text) for text in texts]

word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=200, window=5, min_count=1, workers=4)
embeddings = []
counter=0
for tokenized_text in tokenized_texts:
    embedding = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word
                 in tokenized_text]
    embedding = np.mean(embedding, axis=0)

    embeddings.append(embedding)
    counter+=1
embeddings_array = np.array(embeddings)
output_data = {'model': word2vec_model, 'embeddings': embeddings_array}
with open('D:/word2vec_train_bird_test_300.pickle', 'wb') as file:
    pickle.dump(output_data, file)
    print("finish")


with open('D:/word2vec_train_bird_test_300.pickle', 'rb') as file:
    loaded_data = pickle.load(file)

embeddings = loaded_data['embeddings']

'''file_path = 'D:/output_train_bird.txt'
output_path = 'D:/bert_train_bird_test_200_bert.pickle'  # Adjust output path

bert_model_name = "google-bert/bert-base-uncased"

try:
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
  model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")
  print("Model downloaded successfully!")
except Exception as e:
  print(f"Error downloading model: {e}")



def sentence_to_embedding(sentence):

  encoded_input = tokenizer(sentence, return_tensors="tf", padding="max_length", truncation=True)

  output = model(encoded_input)

  bert_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding

  dense_layer = Dense(200, activation='relu')  # Replace activation if needed
  embedding_reduced = dense_layer(bert_embedding)

  embedding_reduced = embedding_reduced.numpy()

  return embedding_reduced

with open(file_path, 'r') as file:
    lines = file.readlines()
texts = []
counter=0
for line in lines:
    parts = line.split('|')
    if len(parts) > 1:
        text = parts[1].strip()
        if text:
            texts.append(text)

embeddings = []
for text in texts:
    embedding = sentence_to_embedding(text)
    embeddings.append(embedding)
    counter += 1

embeddings_array = np.array(embeddings)

output_data = {'model': None, 'embeddings': embeddings_array}  

with open(output_path, 'wb') as file:
    pickle.dump(output_data, file)
    print("finish")


with open('D:/word2vec_train_bird_test_200_bert.pickle', 'rb') as file:
    loaded_data = pickle.load(file)

embeddings = loaded_data['embeddings']'''