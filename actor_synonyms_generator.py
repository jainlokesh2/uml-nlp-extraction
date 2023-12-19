# actor_synonyms_generator.py
import os
import re
import textwrap
from evaluation import evaluate_classification_performance, simulate_classification_predictions
from util import read_file, write_file
from synonyms import get_stopwords, get_synonyms
from nltk.tokenize import word_tokenize

class ActorSynonymsGenerator:
    @staticmethod
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

    @staticmethod
    def extract_keywords_from_uml_description(uml_description):
        keywords = word_tokenize(uml_description, language='english')
        return [get_synonyms(keyword) for keyword in keywords if keyword.lower() not in get_stopwords()]

    @staticmethod
    def process_plantuml_files(input_path, output_path):
        all_y_true = []
        all_y_pred = []

        for filename in os.listdir(input_path):
            if filename.endswith(".plantuml"):
                plantuml_file_path = os.path.join(input_path, filename)
                plantuml_code = read_file(plantuml_file_path)

                actor_synonyms_list = ActorSynonymsGenerator.generate_actor_use_case_synonyms(plantuml_code)

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
                keywords = ActorSynonymsGenerator.extract_keywords_from_uml_description(plantuml_code)

                # Simulate classification labels based on keywords
                y_true, y_pred = simulate_classification_predictions(keywords, actor_synonyms_list)

                # Accumulate true and predicted labels
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                print(f"Passage generated and saved to {output_file_path}")

        # Evaluate classification performance for all iterations
        evaluate_classification_performance(all_y_true, all_y_pred)
