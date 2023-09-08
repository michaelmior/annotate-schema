# Schema annotation
[![CI](https://github.com/michaelmior/annotate-schema/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelmior/annotate-schema/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/michaelmior/annotate-schema/main.svg)](https://results.pre-commit.ci/latest/github/michaelmior/annotate-schema/main)

This repository contains scripts which attempt to augment a provided JSON Schema using the power of LLMs.

## Description generation

This works by generating a prompt for each JSON path in the schema and then executing a LLM to generate a description for each attribute.
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

### Schema type

While the input to this script is always a JSON Schema, multiple possible formats can be used to derive descriptions during code generation.
In addition to `jsonschema`, current options include [`zod`](https://zod.dev/) and [`typescript`](https://www.typescriptlang.org/docs/handbook/2/objects.html).

### Models

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

## Naming definitions

The `name_definitions.py` script attempts to generate meaningful names for definitions in a schema provided on standard input.
Since this makes use of infill, currently it only works with Facebook's InCoder models.
For example, given a schema containing the definition below, the name `defn0` will be replaced with `person`.

```json
{
  "definitions": {
    "defn0": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "name": { "type": "string" },
        "age": { "type": "integer" },
        "gender": { "type": "string" },
        "email": { "type": "string" }
      }
    }
  },
  ...
}
```

### Models

Any model which is trained using MLM should work here.
In addition, Facebook's InCoder models are supported.

- facebook/incoder-1B
- facebook/incoder-6B
- huggingface/CodeBERTa-small-v1
- microsoft/codebert-base
- microsoft/codebert-base-mlm
- neulab/codebert-javascript

## Selecting relevant keywords

When discovering a schema from data, it's possible to generate keywords such as `minLength` for all string properties.
However, not all of those properties are necessarily relevant for inclusion into the final schema and may just be overfit to the dataset.
To solve this problem, you can train a model on real-world schemas to predict whether a keyword should be included.

```bash
# Download the schemas from JSON Schema Store
$ ./download_schemas.sh

# Extract training data from the
$ pipenv run python extract_keywords.py > extracted.json

# Train the model
pipenv run python embed_training.py extracted.json
```
