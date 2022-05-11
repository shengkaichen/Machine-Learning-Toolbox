from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import pandas as pd
import io


def decision_trees():
    return print("0")




raw_dataset = pd.read_csv(io.BytesIO(uploaded['bank-full.csv']), header = 0, delimiter = ';')
# Dataset is now stored in a Pandas Dataframe
raw_dataset.head()

raw_dataset.shape

#Lets do some feature selection
feature_names = ['age', 'balance', 'duration']
X = raw_dataset[feature_names] #Selected features
y = raw_dataset.y #Target feature (i.e., class label)

#The full dataset is randomly divided into the "train" and "test" sets - This can be extended to train, validate nad test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) #80% training and 20% test

#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
#Train a Decision Tree model
clf = clf.fit(X_train,y_train)
#Use the model to predict the response for our test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Lets visualize the tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names, class_names=['no','yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('bank_decision_tree.png')
Image(graph.create_png())




from gensim import models
from gensim import corpora
from gensim import similarities
docA = "The car is driven on the road"
docB = "The truck is driven on the highway"
docC = 'The capital of Canada is Ottawa and Ottawa has tulips'
stoplist = set('for a of the and to in'.split())
texts = [[word for word in docA.lower().split() if word not in stoplist] for doc in docA]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]
lsi_model.print_topics(2)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
response = tfidf.fit_transform([docA, docB])
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])

# https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/
# https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d
# https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/

