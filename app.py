from anthropic import Anthropic
from flask import Flask, request, jsonify, render_template
import os
from logging.config import dictConfig
import textwrap
import time

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
    }
})

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

system_prompt = textwrap.dedent("""
    Your task is to generate a structured query for the given natural language question about a database table.

    For the structured query, we will define a new query language called TQL (token-based query language).
    The query consists of a sequence of phrases, and each phrase is a sequence of tokens.
    A token can be a column (always enclosed in []), a reserved keyword, an operator, or a value.

    E.g.,
    Question: "How many users bought products worth more than $100 in the last month?"
    TQL: count [user id] [order amount] > 100 [order date] = 'last month'
    Phrases:
      - count [user id]
      - [order amount] > 100
      - [order date] = 'last month'
    Columns: user id, order amount, order date.
    Keywords: count, last month
    Operators: >, =

    You will be given:
      - The database table schema
      - Full reference for TQL keywords with examples
      - A natural language question
""")

with open('keyword_reference.txt', 'r') as f:
    keyword_reference = f.read()

with open('schema.csv', 'r') as f:
    schema = f.read()

@app.route('/chat', methods=['POST'])
def chat():
    # Get the question from the request body
    question = request.json.get('question', '')

    # Make the API call to get a completion
    start_time = time.time()
    completion = client.beta.prompt_caching.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=[
            {
                "type": "text",
                "text": system_prompt,
            },
            {
                "type": "text",
                "text": "<schema>" + schema + "</schema>",
            },
            {
                "type": "text",
                "text": "<reference>" + keyword_reference + "</reference>",
                "cache_control": {
                    "type": "ephemeral",
                }
            },
            {
                "type": "text",
                "text": "Baseed on the above information, generate a syntactically correct TQL query for the given question. Only output a string that is the TQL query.",
            }
        ],
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    end_time = time.time()

    app.logger.info('Completion: %s', completion)
    app.logger.info('Usage: %s', completion.usage)
    app.logger.info('Time taken: %s seconds', end_time - start_time)

    # Return the response as JSON
    response_txt = '\n'.join([c.text for c in completion.content])
    return jsonify({"response": response_txt})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5151)

# ... existing code ...
