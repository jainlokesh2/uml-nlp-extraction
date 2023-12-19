# evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def simulate_classification_predictions(keywords, actor_synonyms_list):
    y_true = [1 if any(keyword.lower() in actions for _, actions, _ in actor_synonyms_list)
              else 0 for keyword in keywords]

    y_pred = [1 if any(keyword.lower() in synonyms for _, _, synonyms in actor_synonyms_list)
              else 0 for keyword in keywords]

    return y_true, y_pred

def evaluate_classification_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
