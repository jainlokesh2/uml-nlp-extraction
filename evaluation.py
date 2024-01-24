from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy
import sys
import subprocess

class Evaluation:
    def __init__(self):
        self.nlp = self.load_spacy_model('en_core_web_md')
        self.scores = []

    def load_spacy_model(self, model_name):
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name)  # Load the model after installation
        return nlp

    def calculate_similarity_scores(self, designed_sentences, created_sentences):
        similarity_scores = []
        for designed, created in zip(designed_sentences, created_sentences):
            designed_doc = self.nlp(designed)
            created_doc = self.nlp(created)
            similarity = designed_doc.similarity(created_doc)
            similarity_scores.append(similarity)
            #print(f"Similarity between '{designed}' and '{created}': {similarity:.2f}")
        return similarity_scores

    def evaluate_performance(self, designed_sentences, created_sentences, thresholds=(0.7, 0.8)):
        similarity_scores = self.calculate_similarity_scores(designed_sentences, created_sentences)
        y_pred = [2 if score >= thresholds[1] else 1 if score >= thresholds[0] else 0 for score in similarity_scores]
        y_true = [2] * len(designed_sentences)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.scores.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    def get_average_scores(self):
        avg_accuracy = sum(score['accuracy'] for score in self.scores) / len(self.scores)
        avg_precision = sum(score['precision'] for score in self.scores) / len(self.scores)
        avg_recall = sum(score['recall'] for score in self.scores) / len(self.scores)
        avg_f1 = sum(score['f1'] for score in self.scores) / len(self.scores)
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }