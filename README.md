# Hostalky demo AI pipeline

Run the following command to install the necessary Python libraries:
- pip install openai openai-agents "fastapi[standard]"

Go to the OpenAI platform and generate your API key:
- https://platform.openai.com/home

Create a file to store your API key:
- touch ./.env

Replace {your API key} with your actual key:
- echo "export OPENAI_API_KEY={your API key}" > ./.env