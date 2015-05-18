import os, re, nltk
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Set the current directory
curr_dir = "/Users/BradLi/Documents/Data Science/NLP/News Classification/topic_corpus"
# Read cleaned texts from CSV files
cleaned_from_file = False
# Read training data from CSV files
train_from_file = False


def clean_data(df, remove_stopwords=True, stemming=True, extract_noun=False):
    """
    Take the raw pandas dataframe as input
    Return the dataframe with cleaned texts appended in the new column "text"
    When extract_noun = True, a new column "noun" will also be appended, which consists of only nouns
    """

    print "Shuffling the data"
    df = shuffle(df)

    print "Cleaning the data"
    # Drop the unwanted columns and keep the rest for training
    df = df.drop(["Serial Number", "Unnamed: 7", "link", "Date", "Source"], axis=1)
    df.columns = map(str.lower, df.columns)

    # Strip the whitespaces
    for col in df.columns:
        df[col] = df[col].apply(str.strip)

    # Remove all non-alphabet characters
    df["title"] = df["title"].str.replace(r"[^a-zA-Z\s-]", r"")
    df["description"] = df["description"].str.replace(r"[^a-zA-Z\s-]", r"")

    if (remove_stopwords == True) or (stemming == True) or (extract_noun == True):

        # Split the sentences into words
        print "Tokenizing the words"
        df["title"] = df["title"].apply(nltk.word_tokenize)
        df["description"] = df["description"].apply(nltk.word_tokenize)

        # Regular expression tokenizer. Disabled as performance is similar to nltk.word_tokenizer
        # pat = r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b"
        # df["title"] = df["title"].apply(lambda x: nltk.regexp_tokenize(x, pat))
        # df["description"] = df["description"].apply(lambda x: nltk.regexp_tokenize(x, pat))

        # Combine news titles and content into one column
        df["text"] = df["title"] + df["description"]

        if extract_noun == True:
            print "Tagging parts-of-speech"
            df["noun"] = df["title"] + df["description"]
            df["noun"] = df["noun"].apply(lambda x: [word[0] for word in nltk.pos_tag(x)
                                                        if word[1] in {"NN", "NNS", "NNP", "NNPS"}])
            df["noun"] = df["noun"].apply(lambda x: " ".join(x))

        if remove_stopwords == True:
            # Convert stop words to sets as sets have constant look-up time and thus faster
            stops = set(stopwords.words("english"))
            print "Removing stop words"
            df["text"] = df["text"].apply(lambda x: [word for word in x if word not in stops])

        if stemming == True:
            # Initialize the Snowball stemmer
            stemmer = SnowballStemmer("english")
            print "Stemming English words"
            df["text"] = df["text"].apply(lambda x: [stemmer.stem(word) for word in x])

            # Lemmatization is currently not applied
            # lemmatizer = WordNetLemmatizer()
            # df["text"] = df["text"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

        df["text"] = df["text"].apply(lambda x: " ".join(x))

    else:
        df["text"] = df["title"] + " " + df["description"]

    return df


def extract_features(series, method, dict_size=50000, bigram=True):
    """
    Take the pandas series as input, where the elements in the series are strings
    Return the feature vector as numpy array
    """

    # TF-IDF weighting
    if method == "TFIDF":
        vectorizer = TfidfVectorizer(analyzer="word", max_features=dict_size,
                                     ngram_range=(1, (2 if bigram == True else 1)),
                                     sublinear_tf=True)

    # Binary or integer count vectorization
    else:
        vectorizer = CountVectorizer(analyzer="word", max_features=dict_size,
                                     ngram_range=(1, (2 if bigram == True else 1)),
                                     binary=(method == "Binary"))

    print "Vectorizing the texts"
    vec = vectorizer.fit_transform(series)
    vec = vec.toarray()
    # test = vectorizer.transform(series)
    # test = test.toarray()

    return vec


def reduce_dim(vec, num_dim, method, label=None):
    """
    Dimension reduction. Two approaches are provided.
    SVD: Truncated SVD maps feature vectors into different subspaces.
    chi2: Chi-square independence test examine the pairwise dependence of features and labels
    """

    print "Performing dimension reduction"

    # Reduce the dimensions using truncated SVD or Chi-Square independence test
    if method == "SVD":
        svd = TruncatedSVD(n_components=num_dim)
        vec = svd.fit_transform(vec)
        # test = svd.transform(vec)
    elif method == "chi2" or method == "f_classif":
        fselect = SelectKBest((chi2 if method == "chi2" else f_classif), k=num_dim)
        vec = fselect.fit_transform(vec, label)
        # test = fselect.transform(vec)

    return vec


os.chdir(curr_dir)

# Read cleaned texts from CSV file
if cleaned_from_file == True:
    cleaned = pd.read_csv("cleaned.csv")
else:
    raw = pd.read_csv("corpus.csv", header=0)
    cleaned = clean_data(df=raw, remove_stopwords=True, stemming=True, extract_noun=False)

# Read training data from CSV file
if train_from_file == True:
    train_data = pd.read_csv("train.csv", header=None)
else:
    train_data = extract_features(series=cleaned.text, method="TFIDF", dict_size=50000, bigram=True)
    train_data = reduce_dim(vec=train_data, num_dim=10000, method="chi2", label=cleaned.category)

# Grid search for model parameters
# param = {"C": [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75]}
# model = LinearSVC(dual=False)
# model = GridSearchCV(model, param, cv=10)
# print "Grid-searching the parameters"
# model.fit(train_data, cleaned.category)
# print "Optimized parameters =", model.best_params_
# print "Best CV score =", model.best_score_
# print "All CV scores =", model.grid_scores_

# Perform cross validation
model = LinearSVC(C=0.75, dual=False)
print "Cross-validating the model"
cv_score = cross_val_score(model, train_data, cleaned.category, cv=5)
print "Cross validation score =", cv_score.mean()

# Fit the model and predict
# model = model.fit(train_data, cleaned.category)
# pred = model.predict(test_data)


########################      Performance Report      ########################

### Default setting:
# dim = 1000

### Naive Bayes
# Gaussian NB, tf-idf vec, unigram: 0.6789

# Bernoulli NB, binary vec, bigram/unigram: 0.7719
# Bernoulli NB, binary vec, unigram: 0.7940

# Multinomial NB, int vec, bigram/unigram: 0.7884
# Multinomial NB, int vec, unigram: 0.8021
# Multinomial NB, int vec, unigram, remove stopwords: 0.8032
# Multinomial NB, int vec, unigram, remove stopwords, stemming: 0.8146
# Multinomial NB, int vec, unigram, remove stopwords, stemming, only nouns: 0.8068
# Multinomial NB, same as above but separate titles and content when vectorizing: 0.7861

### In the following we assume unigrams, stopwords removal and stemming if not otherwise stated

# RF, dim=1000, int vec, n_estimators=100: 0.7720

# Assume max_features = sqrt for Gradient Boosting Machine (GBM)
# RF, dim=1000, int vec, n_estimators=100, n_data=10000: 0.7090
# Bagging tree, dim=1000, int vec, n_estimators=100, n_data=10000: 0.6991
# GBM, dim=1000, int vec, n_estimators=100, n_data=10000, subsample=0.5, max_depth=7: 0.7915

# GBM, dim=5000, int vec, n_estimators=50, n_data=all, subsample=0.5, max_depth=6: 0.7626
# GBM, dim=1000, int vec, n_estimators=50, n_data=all, subsample=0.5, max_depth=15: 0.8079
# GBM, dim=1000, int vec, n_estimators=100, n_data=all, subsample=0.5, max_depth=7: 0.8125
# GBM, dim=1000, int vec, n_estimators=100, n_data=all, subsample=0.5, max_depth=10: 0.8133
# GBM, dim=1000, int vec, n_estimators=100, n_data=all, subsample=0.5, max_depth=12: 0.8150
# GBM, dim=3000, int vec, n_estimators=100, n_data=all, subsample=0.5, max_depth=7: 0.8146
# GBM, dim=1000, int vec, n_estimators=200, n_data=all, subsample=0.3, max_depth=7: 0.8141
# GBM, dim=1000, int vec, n_estimators=200, n_data=all, subsample=0.5, max_depth=7: 0.8170

# Bagging decision trees, int vec, n_estimators=10: 0.7281
# Bagging decision trees, int vec, max_depth=15, n_estimators=10: 0.4073
# Bagging decision trees, int vec, min_samples_split=8, n_estimators=10: 0.7481


# Grid search result for gradient boosting (max_features=None):
# Use 10000 samples, 5-fold CV to speed up training

# mean: 0.56890, std: 0.01091, params: {'n_estimators': 10, 'subsample': 1.0, 'max_depth': 3}
# mean: 0.57670, std: 0.00924, params: {'n_estimators': 10, 'subsample': 0.8, 'max_depth': 3}
# mean: 0.57640, std: 0.00882, params: {'n_estimators': 10, 'subsample': 0.6, 'max_depth': 3}
# mean: 0.60580, std: 0.01231, params: {'n_estimators': 15, 'subsample': 1.0, 'max_depth': 3}
# mean: 0.60590, std: 0.01051, params: {'n_estimators': 15, 'subsample': 0.8, 'max_depth': 3}
# mean: 0.61500, std: 0.00822, params: {'n_estimators': 15, 'subsample': 0.6, 'max_depth': 3}
# mean: 0.63280, std: 0.00891, params: {'n_estimators': 20, 'subsample': 1.0, 'max_depth': 3}
# mean: 0.63470, std: 0.01009, params: {'n_estimators': 20, 'subsample': 0.8, 'max_depth': 3}
# mean: 0.64120, std: 0.00961, params: {'n_estimators': 20, 'subsample': 0.6, 'max_depth': 3}
# mean: 0.62740, std: 0.00831, params: {'n_estimators': 10, 'subsample': 1.0, 'max_depth': 5}
# mean: 0.63290, std: 0.00806, params: {'n_estimators': 10, 'subsample': 0.8, 'max_depth': 5}
# mean: 0.63760, std: 0.00718, params: {'n_estimators': 10, 'subsample': 0.6, 'max_depth': 5}
# mean: 0.65980, std: 0.00525, params: {'n_estimators': 15, 'subsample': 1.0, 'max_depth': 5}
# mean: 0.66260, std: 0.00725, params: {'n_estimators': 15, 'subsample': 0.8, 'max_depth': 5}
# mean: 0.66940, std: 0.01046, params: {'n_estimators': 15, 'subsample': 0.6, 'max_depth': 5}
# mean: 0.68200, std: 0.00684, params: {'n_estimators': 20, 'subsample': 1.0, 'max_depth': 5}
# mean: 0.68770, std: 0.00615, params: {'n_estimators': 20, 'subsample': 0.8, 'max_depth': 5}
# mean: 0.69040, std: 0.00547, params: {'n_estimators': 20, 'subsample': 0.6, 'max_depth': 5}
# mean: 0.66150, std: 0.00707, params: {'n_estimators': 10, 'subsample': 1.0, 'max_depth': 7}
# mean: 0.67270, std: 0.00616, params: {'n_estimators': 10, 'subsample': 0.8, 'max_depth': 7}
# mean: 0.67520, std: 0.00915, params: {'n_estimators': 10, 'subsample': 0.6, 'max_depth': 7}
# mean: 0.69050, std: 0.00614, params: {'n_estimators': 15, 'subsample': 1.0, 'max_depth': 7}
# mean: 0.69840, std: 0.00537, params: {'n_estimators': 15, 'subsample': 0.8, 'max_depth': 7}
# mean: 0.70060, std: 0.00506, params: {'n_estimators': 15, 'subsample': 0.6, 'max_depth': 7}
# mean: 0.70940, std: 0.00580, params: {'n_estimators': 20, 'subsample': 1.0, 'max_depth': 7}
# mean: 0.71400, std: 0.00459, params: {'n_estimators': 20, 'subsample': 0.8, 'max_depth': 7}
# mean: 0.72050, std: 0.00562, params: {'n_estimators': 20, 'subsample': 0.6, 'max_depth': 7} (best result)


### Linear SVM: dim=2000/5000 (unigram/bigram), C=1.0, (0,1) scale, 5-fold CV:
# Remove stopwords, stemming, unigram, int vec: 0.8432
# Remove stopwords, stemming, bigram, int vec: 0.8491
# Remove stopwords, stemming, unigram, TFIDF vec: 0.8463
# Remove stopwords, stemming, bigram, TFIDF vec: 0.8544
# Remove stopwords, stemming, bigram, TFIDF vec, don't scale: 0.8655 (optimal setting)

# Remove stopwords, bigram, int vec: 0.8470
# Remove stopwords, bigram, TFIDF vec: 0.8505

# Don't remove stopwords, bigram, TFIDF vec: 0.8474

# Use above optimal setting but use 500/4500 features for noun-only/all parts-of-speech: 0.8632

# Coarse grid search results with optimal setting, dim=10000 and 5-fold CV:

# mean: 0.25120, std: 0.00007, params: {'C': 0.000244140625}
# mean: 0.36480, std: 0.00298, params: {'C': 0.0009765625}
# mean: 0.62213, std: 0.00219, params: {'C': 0.00390625}
# mean: 0.79938, std: 0.00419, params: {'C': 0.015625}
# mean: 0.84990, std: 0.00193, params: {'C': 0.0625}
# mean: 0.87121, std: 0.00163, params: {'C': 0.25}
# mean: 0.87495, std: 0.00310, params: {'C': 1}
# mean: 0.86499, std: 0.00283, params: {'C': 4}
# mean: 0.84468, std: 0.00365, params: {'C': 16}
# mean: 0.82419, std: 0.00334, params: {'C': 64}

# Fine grid search results with optimal setting, dim=20000 and 10-fold CV:

# mean: 0.87876, std: 0.00340, params: {'C': 0.5}
# mean: 0.87906, std: 0.00347, params: {'C': 0.75} (best result)
# mean: 0.87811, std: 0.00304, params: {'C': 1}
# mean: 0.87732, std: 0.00337, params: {'C': 1.25}
# mean: 0.87615, std: 0.00337, params: {'C': 1.5}
# mean: 0.87440, std: 0.00297, params: {'C': 1.75}
# mean: 0.87305, std: 0.00303, params: {'C': 2}
# mean: 0.87183, std: 0.00285, params: {'C': 2.25}
# mean: 0.87078, std: 0.00273, params: {'C': 2.5}
# mean: 0.86980, std: 0.00250, params: {'C': 2.75}
# mean: 0.86864, std: 0.00248, params: {'C': 3}
# mean: 0.86756, std: 0.00285, params: {'C': 3.25}
# mean: 0.86704, std: 0.00303, params: {'C': 3.5}
# mean: 0.86591, std: 0.00338, params: {'C': 3.75}

########################      Remark      ########################

# P(x_i|y) is not Gaussian, so Gaussian NB is not suitable in this case

# RF makes the split based on randomly selected features which are very sparse, so it yields worse performance
# Bagging trees make splits based on all features
# As max_features gets higher, the randomness between trees is reduced
# and it leads to slightly worse performance than RF (~ 1%)

# We also grid-search max_depth for bagging trees
# and find that performance greatly degraded when max_depth gets lower

# Bagging reduces variance for high-variance models,
# so it does not improve performance if we apply bagging to low-variance models like naive Bayes

# For gradient boosting, lower subsample (a.k.a., stochastic gradient boosting) prevents from overfitting
# It's better to fully grow the trees (consistent with results in bagging trees)
# It's better to have higher n_estimators
# If we train gradient boosting machine with max_features = sqrt, performance will get slightly worse
# but greatly shorten training time
# Number of features does not have hugh impacts on performance as max_depth, n_estimators do

# Linear SVM with bigram/TFIDF yields the best performance among all models

# Stop words bear no information here so we remove them

# Titles are too short to bear accurate information for classification
# Separating titles and content does not work

# Including an additional bag-of-nouns in addition to original features does not improve accuracy

