import json

input_file = "../SQuAD_Datasets/train-v1.1.json"
output_file = "../SQuAD_Datasets/train-v1.1-2000_to_3000.json"

start_question_count = 2000
end_question_count = 3000

with open(input_file, 'r') as f:
    squad_data = json.load(f)

small_data = {"data": []}
question_count = 0
current_question_index = 0

for article in squad_data["data"]:
    new_article = {"title": article["title"], "paragraphs": []}
    for paragraph in article["paragraphs"]:
        new_paragraph = {"context": paragraph["context"], "qas": []}
        for qa in paragraph["qas"]:
            current_question_index += 1
            if current_question_index > end_question_count:
                break
            if current_question_index > start_question_count:
                new_paragraph["qas"].append(qa)
                question_count += 1
        if new_paragraph["qas"]:
            new_article["paragraphs"].append(new_paragraph)
        if current_question_index > end_question_count:
            break
    if new_article["paragraphs"]:
        small_data["data"].append(new_article)
    if current_question_index > end_question_count:
        break

with open(output_file, 'w') as f:
    json.dump(small_data, f, indent=2)

print(f"Small dataset created with questions {start_question_count + 1} to {end_question_count}.")
