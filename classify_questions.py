"""
classify_questions - trains and runs a model to classify questions
"""
import json
import pickle
from typing import *

from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def read_labelled_data(fname: str) -> List[Dict]:
    """
    Reads the data labelled from an earlier clustering operation.
    :param fname: full path and filename of the file to read
    :return: list of dicts
    """
    with open(fname, 'r') as fidin:
        return json.load(fidin)


def train_run_classifier(labelled_data: List[Dict], sbert_model: str = "all-MiniLM-l6-v2"):
    """
    Trains a question classifier and saves predictions for entire dataset.
    :param labelled_data: clustered questions
    :param sbert_model: SentenceTransformers model for generating embeddings (defaults to "all-MiniLM-l6-v2")
    :return: None
    """
    train, test = train_test_split(labelled_data, test_size=0.1)
    print(f"{len(train)} samples train, {len(test)} samples test")

    cluster_model = SentenceTransformer(sbert_model)
    print("Generating embeddings...")
    X_train = cluster_model.encode(
        [sample['question'] for sample in train],
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    X_test = cluster_model.encode(
        [sample['question'] for sample in test],
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    y_train = [sample['cluster'] for sample in train]
    y_test = [sample['cluster'] for sample in test]

    print("Training question classifier")
    clf = KNeighborsClassifier(
        weights='distance',
        algorithm='auto'
    )
    clf.fit(X_train, y_train)

    print("Testing classifier on holdout set")
    y_pred = clf.predict(X_test)
    print(f"Macro F1: {f1_score(y_true=y_test, y_pred=y_pred, average='macro'):.3f}")
    print(f"Categorical Metrics:\n{classification_report(y_true=y_test, y_pred=y_pred)}")

    print("\nSaving model")
    with open("./output/question_classifier.pkl", "wb") as fidout:
        pickle.dump(clf, fidout)
    # Verify save worked
    print("Verifying saved model with reload")
    with open("./output/question_classifier.pkl", "rb") as fidin:
        clf = pickle.load(fidin)

    print("\nGenerating predictions for entire dataset")
    final_data = list()
    predicted_train_labels = clf.predict(X_train)
    for i in range(len(train)):
        sample = train[i]
        sample['predicted_cluster'] = predicted_train_labels[i]
        final_data.append(sample)
    for i in range(len(test)):
        sample = test[i]
        sample['predicted_cluster'] = predicted_train_labels[i]
        final_data.append(sample)
    with open('./output/predicted_question_labels.json', 'w') as fidout:
        json.dump(final_data, fidout, indent=2)
    print(f"Predictions saved to './output/predicted_question_labels.json'")

def main():
    labelled_data = read_labelled_data("./output/clustered_questions.json")
    train_run_classifier(labelled_data)


if __name__ == "__main__":
    main()

