# cluster_questions

Once upon a time I was asked to come up with a way to classify user questions. There are any number of ways to do it, but I've always liked [SentenceTransformers](https://sbert.net/index.html) for this kind of thing.  In particular I'm a big fan of [sbert's fast clustering](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py), because not only does it live up to its name but it also offers niceties like returning the clusters sorted largest:smallest, the first element in each cluster is the centroid, and so on.

Here's an example of the approach I usually take:

1. Use SentenceTransformers to cluster the input texts by cosine similarity.
2. The centroids can be considered as the cluster themes or in this case the class labels.
3. Train a classifier on this now-labelled data.
4. _Optional_. Create a label map that maps centroids to more user-friendly and/or understandable labels to use in a production scenario e.g.

```python
LABEL_MAP = {  # Centroid: Production Label
    "A flashing red traffic light signifies that a driver should do what?": "Questions from a driver's test",
    "A pita is a type of what?": "Questions about food",
}
```

You could of course manually relabel the centroids before you train the classifier, or use an LLM, etc., but I like to keep the centroids as-is if for no other reason than I know they will be semantically similar to the questions. If I manually change the labels, I think it's at least theoretically possible that I'd also be changing cosine similarities between the new label and the cluster constituents.

This repo contains a simplified version of my usual approach.  `cluster_questions.py` does the initial clustering and creates the labelled training data used to train the classifier in `classify_questions.py`.  I can't share the original questions, but I've included a sample question file to give the basic idea of how it all works. ([source](https://gist.github.com/cmota/f7919cd962a061126effb2d7118bec72))