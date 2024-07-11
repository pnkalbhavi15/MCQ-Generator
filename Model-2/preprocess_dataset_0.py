import json

input_file = "../SQuAD_Datasets/train-v1.1.json"
output_file = "../SQuAD_Datasets/train-v1.1-0_to_1000.json"

with open(input_file, 'r') as f:
    squad_data = json.load(f)

small_data = {"data": []}
question_count = 0

for article in squad_data["data"]:
    new_article = {"title": article["title"], "paragraphs": []}
    for paragraph in article["paragraphs"]:
        new_paragraph = {"context": paragraph["context"], "qas": []}
        for qa in paragraph["qas"]:
            if question_count >= 1000:
                break
            new_paragraph["qas"].append(qa)
            question_count += 1
        if new_paragraph["qas"]:
            new_article["paragraphs"].append(new_paragraph)
        if question_count >= 1000:
            break
    if new_article["paragraphs"]:
        small_data["data"].append(new_article)
    if question_count >= 1000:
        break

with open(output_file, 'w') as f:
    json.dump(small_data, f, indent=2)

print(f"Small dataset created with the first {question_count} questions.")
