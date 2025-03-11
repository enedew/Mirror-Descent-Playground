import json

with open('base_experiment.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)