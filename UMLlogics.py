import os
import re
import nltk
import textwrap
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.metrics import precision_score, recall_score, f1_score

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
        synonyms = [get_synonyms(keyword) for keyword in keywords if keyword.lower() not in ["to", "from", "no"]]

        actor_synonyms_list.append([actor_names, use_case[0], synonyms])

    return actor_synonyms_list

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def evaluate_nlp_classification(y_true, y_pred):
    # Evaluate NLP-based classification metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"NLP Precision: {precision:.4f}")
    print(f"NLP Recall: {recall:.4f}")
    print(f"NLP F1 Score: {f1:.4f}")

def process_plantuml_files(input_path, output_path):
    for filename in os.listdir(input_path):
        if filename.endswith("userLogin.plantuml"):
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

            # Replace this with your actual NLP-based extracted requirements
            nlp_extracted_requirements = ["The system must handle user logins securely.",
                                          "The software should support different user roles.",
                                          "The application might allow file uploads."]

            # Simulate NLP-based classification labels
            y_true = [1 if 'important' in actions else 0 for _, actions, _ in actor_synonyms_list]
            y_pred = [1 if 'critical' in synonyms else 0 for _, _, synonyms in actor_synonyms_list]

            # Evaluate NLP-based classification performance
            evaluate_nlp_classification(y_true, y_pred)

            print(f"Passage generated and saved to {output_file_path}")

# Provide the input and output paths for evaluation
evaluation_input_directory = '.'
evaluation_output_directory = '.'

# Process PlantUML files and evaluate NLP-based classification performance
process_plantuml_files(evaluation_input_directory, evaluation_output_directory)
