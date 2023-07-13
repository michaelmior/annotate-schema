# Schema annotation

This repository contains a single script which attempts to augment a provided schema to add a `description` to each attribute using the power of LLMs.
It works by generating a prompt for each JSON path in the schema and then executing a LLM to generate a description for each attribute.


For example, the schema below has a single property `foo`.

    {
      "type": "object",
      "properties": {
        "foo": {
          "type": "string"
        }
      }
    }

For this, we generate a prompt like the following:

    {
      "type": "object",
      "properties": {
        "foo": {
          "type": "string",
          "description": "

Note that the prompt ends after the description is started.
Generation continues until an unescaped closing quote is encountered.
To run with your own schema, provide it on standard input.
The resulting schema with descriptions is written to standard output.

    pipenv run python annotate_schema.py < input_schema.json
