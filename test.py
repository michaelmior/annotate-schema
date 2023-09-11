import unittest

import jsonpath_ng
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import annotate_schema


class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM"
        )
        self.schema = {"type": "string"}

    def test_gen_description(self):
        AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM"
        )
        path = jsonpath_ng.parse("$")
        desc = annotate_schema.generate_description(
            self.schema,
            path,
            "jsonschema",
            self.model,
            self.tokenizer,
        )
        self.assertIsInstance(desc, str)

    def test_convert_jsonschema(self):
        self.assertEqual(
            '{"type": "string"}',
            annotate_schema.convert_schema(self.schema, "jsonschema"),
        )

    def test_convert_pydantic(self):
        self.assertIn(
            "class Model(BaseModel):\n    __root__: str",
            annotate_schema.convert_schema(self.schema, "pydantic"),
        )

    def test_convert_typescript(self):
        self.assertIn(
            "export type JSONSchema = string;",
            annotate_schema.convert_schema(self.schema, "typescript"),
        )

    def test_convert_zod(self):
        self.assertEqual(
            "const schema = z.string();",
            annotate_schema.convert_schema(self.schema, "zod").strip(),
        )


if __name__ == "__main__":
    unittest.main()
