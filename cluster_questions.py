"""
cluster_questions - clusters questions based on cosine similarity. Cluster centroids can be considered as the class
labels for a subsequent classification; n.b. in a "real" application we'd probably have a centroid:class name map to
return more intuitive classes. Keeping cluster names separate from "friendly" names makes it easier to update consumer
services w/o retraining.

e.g. {
    "A flashing red traffic light signifies that a driver should do what?": "Questions from a driver's test",
    "A pita is a type of what?": "Questions about food",
    ...
    }
"""
import json
from typing import *
from sentence_transformers import SentenceTransformer, util



def read_question_file(fname: str) -> List[Dict]:
    """
    Reads the sample questions JSON file
    :param fname: full path and filename of the file
    :return: list of dicts
    """
    with open(fname, 'r') as fidin:
        return json.load(fidin)


def gen_question_embeddings(
        questions: List[Dict],
        sbert_model: str = "all-MiniLM-l6-v2"
) -> "torch.Tensor":
    """
    Generates embeddings for a list of question objects.
    :param questions: questions to embed
    :param sbert_model: model to use (defaults to "all-MiniLM-l6-v2")
    :return: embeddings tensor
    """
    print("Generating question embeddings...")
    cluster_model = SentenceTransformer(sbert_model)
    return cluster_model.encode(
        [question['question'] for question in questions],
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True
    )


def cluster_questions(
        questions: List[Dict],
        sbert_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 25,
        sim_threshold: float = 0.75
) -> list:
    """
    Uses "community detection" (cosine similarity) to cluster questions.
    :param questions: questions to cluster
    :param sbert_model: SentenceTransformers model to use for embeddings (defaults to "all-MiniLM-L6-v2")
    :param min_cluster_size: minimum number of questions in a cluster. Clusters below this threshold will be discarded,
    samples will be considered outliers.
    :param sim_threshold: similarity threshold
    :return: list of the clusters found
    """
    question_embeddings = gen_question_embeddings(questions=questions, sbert_model=sbert_model)
    question_clusters = util.community_detection(
        question_embeddings,
        min_community_size=min_cluster_size,
        threshold=sim_threshold
    )
    return question_clusters


def main():
    labelled_data = list()
    questions = read_question_file('./data/questions.json')
    # Normally I start w. cluster size 25-30, sim_threshold 0.75 but found these values to work well here
    clusters = cluster_questions(questions, min_cluster_size=3, sim_threshold=0.65)
    # First element in each cluster is the centroid, which we can take as a general theme or class label
    clustered = list()
    # We'll label outliers as '<< UNKNOWN >>' for now, can always revisit in future versions of the model
    outliers = [i for i in range(len(questions))]
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1}: {len(cluster)} question(s)")
        print(f"Centroid: {questions[cluster[0]]['question']}")
        clustered.extend(cluster)
        for sentence_id in cluster[1:]:
            # print("\t", questions[sentence_id]['question_text'])
            labelled_data.append({'question': questions[sentence_id]['question'], 'cluster': questions[cluster[0]]['question']})
    outliers = list(set(outliers) - set(clustered))
    assert len(outliers) + len(clustered) == len(questions)
    print(f"\nFinal cluster results: {len(clustered)} questions clustered; {len(outliers)} were not clustered")
    for outlier in outliers:
        labelled_data.append({'question': questions[outlier]['question'], 'cluster': '<< UNKNOWN >>'})
    with open('./output/clustered_questions.json', 'w') as fidout:
        json.dump(labelled_data, fidout)


if __name__ == "__main__":
    main()
