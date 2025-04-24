
import math
import pandas as pd


train_data = pd.read_csv('film-genres-train.tsv', sep='\t', header=None, names=[ 'genre','description'])
test_data = pd.read_csv('film-genres-test.tsv', sep='\t', header=None, names=[ 'genre','description'])


stop_words = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
    'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",
    'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
    'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
    "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
    'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've",
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what',
    "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you',
    "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves'
]

 
def clean_stop_words_text(text):
    cleaned_text = ''
    for char in text.lower():
        if char.isalpha() or char.isspace():
            cleaned_text += char
        else:
            cleaned_text += ' '
    return cleaned_text

def tokenize(text): 
    cleaned_text = clean_stop_words_text(text)
    tokens = cleaned_text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


genre_word_counts = {}  # Count of words in each genre
genre_counts = {}       # Count of documents in each genre
vocabulary = set()      # All unique words

for _, row in train_data.iterrows():
    genre = row['genre']
    description = row['description']
    
    if genre not in genre_counts:
        genre_counts[genre] = 0
        genre_word_counts[genre] = {}
    
    genre_counts[genre] += 1
    
    tokens = tokenize(description) 
    for token in tokens:
        if token not in genre_word_counts[genre]:
            genre_word_counts[genre][token] = 0
        genre_word_counts[genre][token] += 1
        vocabulary.add(token) 


total_docs = len(train_data)
prior_probs = {genre: count / total_docs for genre, count in genre_counts.items()}


# Implement Naive Bayes classifier
def predict_genre(description):
    tokens = tokenize(description)
    max_prob = -math.inf
    best_genre = None
    
    for genre in genre_counts:
        log_prob = math.log(prior_probs[genre])
        
        for token in tokens:
            if token in vocabulary:
                # Use Laplace smoothing (add 1)
                word_count = genre_word_counts[genre].get(token, 0) + 1
                total_words = sum(genre_word_counts[genre].values()) + len(vocabulary)
                log_prob += math.log(word_count / total_words)
        
        if log_prob > max_prob:
            max_prob = log_prob
            best_genre = genre
            
    return best_genre


results = {}
for genre in genre_counts:
    results[genre] = {'correct': 0, 'incorrect': 0}

for _, row in test_data.iterrows():
    actual_genre = row['genre']
    description = row['description']
    predicted_genre = predict_genre(description)
    
    if predicted_genre == actual_genre:
        results[actual_genre]['correct'] += 1
    else:
        results[actual_genre]['incorrect'] += 1


print("Genre Classification Results:")
print("-" * 30)
for genre, counts in results.items():
    correct = counts['correct']
    incorrect = counts['incorrect']
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0
    
    print(f"Genre: {genre}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}")
    print("-" * 30)

total_correct = sum(counts['correct'] for counts in results.values())
total_predictions = sum(counts['correct'] + counts['incorrect'] for counts in results.values())
overall_accuracy = total_correct / total_predictions

print(f"Overall accuracy: {overall_accuracy:.4f}")


user_description = input("Enter a movie description for genre prediction: ")
 
predicted_genre = predict_genre(user_description)
print(f"Predicted genre: {predicted_genre}")
 
print("\nConfidence scores by genre:")
tokens = tokenize(user_description)
scores = {}

for genre in genre_counts: 
    log_prob = math.log(prior_probs[genre])
    
    for token in tokens:
        if token in vocabulary:
            word_count = genre_word_counts[genre].get(token, 0) + 1
            total_words = sum(genre_word_counts[genre].values()) + len(vocabulary)
            log_prob += math.log(word_count / total_words)
    
    scores[genre] = log_prob

sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)

for genre, score in sorted_genres:
    print(f"{genre}: {score:.2f}")

