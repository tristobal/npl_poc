import fasttext
import re


def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


# Reviews to check
reviews = [
    "This restaurant literally changed my life. This is the best food I've ever eaten!",
    "I hate this place so much. They were mean to me.",
    "I don't know. It was ok, I guess. Not really sure what to say.",
    "This used to be a giant parking lot where government employees that worked in the country building would park. They moved all the parking underground and built an awesome park here instead. It's literally the reverse of the Joni Mitchell song."
]

# Pre-process the text of each review so it matches the training format
preprocessed_reviews = list(map(strip_formatting, reviews))

# Load the model
# fasttext supervised -input fasttext_dataset_training.txt -output reviews_model_ngrams -wordNgrams 2
#classifier = fasttext.load_model('reviews_model_ngrams.bin')
classifier = fasttext.load_model('reviews_model.bin')

# Get fastText to classify each review with the model
labels, probabilities = classifier.predict(preprocessed_reviews, 1)

# Print the results
for review, label, probability in zip(reviews, labels, probabilities):
    stars = int(label[0][-1])

    print("{} ({}% confidence)".format("☆" * stars, int(probability[0] * 100)))
    print(review)
    print()
