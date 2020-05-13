import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import re
sentences = []
with open("TrainingData_text_NTU.txt","r") as f:
    for line in f:
        line = re.sub(",","",line)
        line = re.sub("\n","",line)
        line = line.split(" ")
        sentences += [line]
# train gensim model
model = Word2Vec(sentences, min_count=1, size=5)

# summarize vocabulary
words = list(model.wv.vocab)

sentence = "{} {}\n".format(str(len(words)),'5')
with open("vec5.txt","a") as f:
    f.write(sentence)
for word in words:
    input = model[word]
    with open("vec5.txt","a") as f:
        f.write(word)
        for inp in input:
            f.write(" "+str(inp))
        f.write('\n')

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig("embedding.png")
tsne_plot(model)
