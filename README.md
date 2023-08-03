# Schema annotation
[![CI](https://github.com/michaelmior/annotate-schema/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelmior/annotate-schema/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/michaelmior/annotate-schema/main.svg)](https://results.pre-commit.ci/latest/github/michaelmior/annotate-schema/main)

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

## Schema type

While the input to this script is always a JSON Schema, multiple possible formats can be used to derive descriptions during code generation.
In addition to `jsonschema`, current options include [`zod`](https://zod.dev/) and [`typescript`](https://www.typescriptlang.org/docs/handbook/2/objects.html).

## Models

Multiple possible models can be used for the generation.
Currently most models which support `AutoModelForCausalLM` and `AutoTokenizer` should work.
The specific model can be specified with the `-m/--model` flag.
Note that some models may not currently support GPU inference.
If a model produces errors, try running again with the `--cpu` flag.
A few examples are given below.

- `bigcode/santacoder`
- `bigcode/starcoder`
- `facebook/incoder-1B`
- `facebook/incoder-6B`
- `replit/replit-code-v1-3b`
- `Salesforce/codegen-350M-mono`
- `Salesforce/codegen-350M-multi`
- `Salesforce/codegen-6B-mono`
- `Salesforce/codegen-6B-multi`
- `Salesforce/codegen-16B-mono`
- `Salesforce/codegen-16B-multi`
- `Salesforce/codegen2-1B`
- `Salesforce/codegen2-7B`
- `Salesforce/codegen2-16B`
- `Salesforce/codegen25-7b-mono`
- `Salesforce/codegen25-7b-multi`
- `TheBloke/Codegen25-7B-mono-GPTQ` (with `--model-basename gptq_model-4bit-128g`)
