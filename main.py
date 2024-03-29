# main.py
from actor_synonyms_generator import ActorSynonymsGenerator

# Provide the input and output paths for evaluation
evaluation_input_directory = './input'
evaluation_output_directory = './output'
evaluation_designed_output_directory = './designedoutput'

# Process PlantUML files and evaluate classification performance
ActorSynonymsGenerator.process_plantuml_files(evaluation_input_directory, evaluation_output_directory,evaluation_designed_output_directory)
