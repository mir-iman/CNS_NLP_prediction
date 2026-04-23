import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import _pickle as pickle
import bz2


class SCARBoW:
    def __init__(self, config, undersample=False):
        self.max_tokens = config.max_tokens
        self.use_idf = config.use_idf

        if undersample:
            self.data_dir = os.path.join(config.data_dir, config.target + "_undersampled")
        else:
            self.data_dir = os.path.join(config.data_dir, config.target)

        self.train_file = os.path.join(self.data_dir, f"train_bow_{self.max_tokens}.csv")
        self.dev_file = os.path.join(self.data_dir, f"dev_bow_{self.max_tokens}.csv")
        self.test_file = os.path.join(self.data_dir, f"test_bow_{self.max_tokens}.csv")

        if not (
            os.path.exists(self.train_file)
            and os.path.exists(self.dev_file)
            and os.path.exists(self.test_file)
        ):
            self.raw_train_data = pd.DataFrame(columns=["label", "text", "vector"])
            self.raw_dev_data = pd.DataFrame(columns=["label", "text", "vector"])
            self.raw_test_data = pd.DataFrame(columns=["label", "text", "vector"])

            self.raw_train_data = self.read_labels_and_tokens("train")
            self.raw_dev_data = self.read_labels_and_tokens("dev")
            self.raw_test_data = self.read_labels_and_tokens("test")

            self.vectorize_tokens()

        self.train_data = pd.read_csv(self.train_file)
        self.dev_data = pd.read_csv(self.dev_file)
        self.test_data = pd.read_csv(self.test_file)

    @staticmethod
    def label_transform(label):
        label = str(label).strip()

        if label in {"0", "1"}:
            return float(label)
        if label == "10":
            return 0.0
        if label == "01":
            return 1.0

        raise ValueError("Invalid target provided, supports '0'/'1' or '10'/'01'")

    def vectorize_tokens(self):
        vectorizer = CountVectorizer(
            max_features=self.max_tokens,
            tokenizer=StemTokenizer(),
            lowercase=True,
        )

        train_counts = vectorizer.fit_transform(self.raw_train_data["text"])
        dev_counts = vectorizer.transform(self.raw_dev_data["text"])
        test_counts = vectorizer.transform(self.raw_test_data["text"])

        vectorizer_filename = os.path.join(self.data_dir, f"vectorizer_{self.max_tokens}.bz2")
        with bz2.BZ2File(vectorizer_filename, "w") as f:
            pickle.dump(vectorizer, f)

        tfidf_transformer = TfidfTransformer(use_idf=self.use_idf).fit(train_counts)
        self.raw_train_data["vector"] = tfidf_transformer.transform(train_counts).todense().tolist()
        self.raw_dev_data["vector"] = tfidf_transformer.transform(dev_counts).todense().tolist()
        self.raw_test_data["vector"] = tfidf_transformer.transform(test_counts).todense().tolist()

        self.raw_train_data.to_csv(self.train_file, index=False)
        self.raw_dev_data.to_csv(self.dev_file, index=False)
        self.raw_test_data.to_csv(self.test_file, index=False)

    def read_labels_and_tokens(self, split):
        filename = os.path.join(self.data_dir, split + ".tsv")

        if split == "train":
            df = self.raw_train_data
        elif split == "dev":
            df = self.raw_dev_data
        elif split == "test":
            df = self.raw_test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        with open(filename, "r", encoding="utf-8") as file:
            for i, line in enumerate(tqdm(file)):
                values = line.rstrip("\n").split("\t", maxsplit=1)
                assert len(values) == 2, "Expected exactly one tab separating label and text"
                label, raw_text = values
                df.loc[i, "label"] = self.label_transform(label)
                df.at[i, "text"] = raw_text

        return df

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def get_test_data(self):
        return self.test_data


class StemTokenizer:
    def __init__(self):
        self.sbs = SnowballStemmer("english", ignore_stopwords=True)
        self.stop_words = set(stopwords.words("english"))

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.sbs.stem(t) for t in tokens if t not in self.stop_words]