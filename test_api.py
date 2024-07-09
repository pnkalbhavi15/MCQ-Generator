import requests
def generate_questions(input_text):
    # Define the API URL
    api_url = "http://localhost:8001/generate_questions/"
    # Define the input data (text for question generation)
    input_data = {"text": input_text}
    # Send POST request to the API endpoint
    response = requests.post(api_url, json=input_data)
    # Check the response status code
    if response.status_code == 200:
        # Print the JSON response (generated questions)
        print("Generated Questions:")
        for idx, question in enumerate(response.json()["questions"], start=1):
            print(f"{idx}. {question}")
    else:
        # Print error message if request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)
# Example input text for question generation
example_text = """
The Apollo program, also known as Project Apollo, was the third United States human
spaceflight program carried out by the National Aeronautics and Space Administration (NASA),
which succeeded in landing the first humans on the Moon from 1969 to 1972.
"""
# Call the function to generate questions
generate_questions(example_text)
