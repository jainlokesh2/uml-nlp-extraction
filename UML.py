import os
import re
import nltk
import textwrap
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from synonyms import get_stopwords

nltk.download('punkt')
nltk.download('wordnet')

def get_synonyms(keyword):
    synsets = wordnet.synsets(keyword)
    return ', '.join(synsets[0].lemma_names()[:3]) if synsets else "No synonyms available."

def generate_actor_use_case_synonyms(plantuml_code):
    use_cases = re.findall(r'usecase "(.*?)" as (\w+)', plantuml_code)
    actor_connections = re.findall(r'(\w+) -- (\w+)', plantuml_code)

    use_case_actors = {use_case[1]: [] for use_case in use_cases}

    for connection in actor_connections:
        actor, use_case = connection
        if use_case in use_case_actors:
            use_case_actors[use_case].append(actor)

    actor_synonyms_list = []

    for use_case in use_cases:
        actors = use_case_actors.get(use_case[1], [])
        actor_names = ', '.join(actors)
        keywords = word_tokenize(use_case[0], language='english')
        synonyms = [get_synonyms(keyword) for keyword in keywords if keyword.lower() not in get_stopwords()]

        actor_synonyms_list.append([actor_names, use_case[0], synonyms])

    return actor_synonyms_list

def extract_keywords_from_uml_description(uml_description):
    keywords = word_tokenize(uml_description, language='english')
    return [get_synonyms(keyword) for keyword in keywords if keyword.lower() not in get_stopwords()]

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

def process_plantuml_files(input_path, output_path):
    all_y_true = []
    all_y_pred = []

    for filename in os.listdir(input_path):
        if filename.endswith(".plantuml"):
            plantuml_file_path = os.path.join(input_path, filename)
            plantuml_code = read_file(plantuml_file_path)

            actor_synonyms_list = generate_actor_use_case_synonyms(plantuml_code)

            passage_lines = []
            for role, actions, synonyms in actor_synonyms_list:
                role_description = f"{role.capitalize()} must  {actions}."
                synonyms_description = f"Synonyms for {actions}: {', '.join(synonyms).lower()}." if synonyms != "No synonyms available." else "No synonyms available."
                passage_lines.extend([role_description])

            passage = "\n".join(passage_lines)
            wrapped_passage = textwrap.fill(passage, width=80)

            output_file_path = os.path.join(output_path, os.path.splitext(filename)[0] + ".txt")
            write_file(output_file_path, wrapped_passage)

            # Extract keywords from the UML description
            keywords = extract_keywords_from_uml_description(plantuml_code)

            # Simulate classification labels based on keywords
            y_true, y_pred = simulate_classification_predictions(keywords, actor_synonyms_list)

            # Accumulate true and predicted labels
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            print(f"Passage generated and saved to {output_file_path}")

    # Evaluate classification performance for all iterations
    evaluate_classification_performance(all_y_true, all_y_pred)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

# Provide the input and output paths for evaluation
evaluation_input_directory = '.'
evaluation_output_directory = '.'

# Process PlantUML files and evaluate classification performance
process_plantuml_files(evaluation_input_directory, evaluation_output_directory)
