import os
import re
import textwrap
from evaluation import Evaluation
from util import read_file, write_file , read_and_process_design_sentences
from synonyms import get_stopwords, get_synonyms
from nltk.tokenize import word_tokenize
from simplenlg.lexicon import Lexicon
from simplenlg.framework import NLGFactory
from simplenlg.realiser import Realiser
from simplenlg.features import Feature

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
    def process_plantuml_files(input_path, output_path, designed_output_path):
        evaluation = Evaluation()
        lexicon = Lexicon.getDefaultLexicon()
        nlgFactory = NLGFactory(lexicon)
        realiser = Realiser(lexicon)

        for filename in os.listdir(input_path):
            if filename.endswith(".plantuml"):
                plantuml_file_path = os.path.join(input_path, filename)
                plantuml_code = read_file(plantuml_file_path)

                actor_synonyms_list = ActorSynonymsGenerator.generate_actor_use_case_synonyms(plantuml_code)

                passage_lines = []
                for role, actions, synonyms in actor_synonyms_list:
                    # Create more descriptive sentences
                    p = nlgFactory.createClause()
                    p.setSubject(role.capitalize())
                    action_parts = actions.lower().split(maxsplit=1)  # This will split at the first space
                    if len(action_parts) == 2:
                        action_verb, action_object = action_parts
                    else:
                        # Default verb to 'do' if no clear verb is given
                        action_verb, action_object = 'do', actions.lower()
                    p.setVerb(action_verb)
                    objectNP = nlgFactory.createNounPhrase("the", action_object)
                    p.setObject(objectNP)
                    p.setFeature(Feature.MODAL, "must")
                    role_description = realiser.realiseSentence(p)

                    synonyms_description = "Synonyms for {}: {}.".format(actions, ', '.join(synonyms).lower()) if synonyms else "No synonyms available."
                    passage_lines.extend([role_description])

                passage = "\n".join(passage_lines)
                output_file_path = os.path.join(output_path, os.path.splitext(filename)[0] + ".txt")
                write_file(output_file_path, passage)

                #print(f"Passage generated and saved to {output_file_path}")

            # Evaluate classification performance for all iterations
            file_path = os.path.join(designed_output_path, os.path.splitext(filename)[0] + "_design.txt")
            designed_sentences = read_and_process_design_sentences(file_path)
            evaluation.evaluate_performance(designed_sentences, passage_lines)
        average_scores = evaluation.get_average_scores()
        print("\nScores:")
        print(f' Accuracy: {average_scores["avg_accuracy"]:.2f}')
        print(f' Precision: {average_scores["avg_precision"]:.2f}')
        print(f' Recall: {average_scores["avg_recall"]:.2f}')
        print(f' F1 Score: {average_scores["avg_f1"]:.2f}')

