#!/bin/sh

mkdir -p schemas/
wget -q -O- https://raw.githubusercontent.com/SchemaStore/schemastore/master/src/api/json/catalog.json | jq -r '.schemas[] | .url' > schemas/schema_urls.txt
(cd schemas/; wget -nc -w 1 -i schema_urls.txt)
