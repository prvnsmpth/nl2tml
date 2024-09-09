import textwrap
import time

class DataGenerator:
    system_prompt = textwrap.dedent("""
    You are given a reference to the Thoughtspot query language (TQL). You are also given a database table schema. 
    Your task is to analyze the columns in the table, and come up with interesting and non-trivial analysis in 
    conversational format, like so:

    Example 1:
    User: total delinquent amount 
    TQL: sum [delinquent amount]
    User: also show the total number of such accounts? Also group by loan status.
    TQL: sum [delinquent amount] sum [delinquent accounts] by [loan status]
    User: limit this to loans with credit pull in last 6 months.
    TQL: sum [delinquent amount] sum [delinquent accounts] by [loan status] [last credit pull date] = 'last 6 months'

    Example 2:
    User: how many loans with grade A or B?
    TQL: count [id] [grade] = 'A' [grade] = 'B'
    User: filter down to only those where borrower's income is greater than $5000 monthly
    TQL: count [id] [grade] = 'A' [grade] = 'B' average [income] > 5000
    User: for these loans, what are the top 5 loan purposes?
    TQL: top 5 [purpose] [id] in ([id] [grade] = 'A' [grade] = 'B' average [income] > 5000)

    As you can see, the user progressively refines the query to get more and more interesting results.
    You must generate more such examples that cover all possible TQL keywords, and also try to cover all the 
    available columns and data types in the table.
                                    
    You must output examples in JSON format. Make each conversation a list of JSON objects, where each JSON object 
    has two fields: (nl_query, tql_query). Output just the examples, no intro, no explanations, no comments. One conversation per line.
    Each conversation should have 3-5 user messages.
    """)

    def __init__(self, anthropic_client, schema, keyword_reference, logger):
        self.anthropic_client = anthropic_client
        self.schema = schema
        self.keyword_reference = keyword_reference
        self.logger = logger

    def generate_examples(self, output_file="data.json", num_examples=10):
        start_time = time.time()
        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                },
                {
                    "type": "text",
                    "text": f"<schema>{self.schema}</schema>",
                },
                {
                    "type": "text",
                    "text": f"<reference>{self.keyword_reference}</reference>",
                    "cache_control": {
                        "type": "ephemeral",
                    }
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""Generate {num_examples} examples."""),
                }
            ]
        )
        end_time = time.time()

        self.logger.info(f"Time taken: {end_time - start_time} seconds")
        self.logger.info(f"Usage: {response.usage}")

        with open(output_file, "a+") as f:
            for content in response.content:
                if content.type == "text":
                    f.write(content.text)
                    f.write("\n")
        