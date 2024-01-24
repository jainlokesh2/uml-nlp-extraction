# util.py
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
        
def read_and_process_design_sentences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Split the content by full stop and strip unnecessary whitespace or newline characters
        sentences = [sentence.strip() + '.' for sentence in content.split('.') if sentence.strip()]
    return sentences