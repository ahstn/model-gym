# String scores for the RunInfra trial

Decision saved on 26 October 2026 after the user approved a 200-image trial. Keep the technical-quality scale at 1–5 in steps of 0.1. Change the wire representation to a string enum containing `"1.00"`, `"1.10"`, through `"5.00"`, plus null. A valid string is decoded with Decimal into the existing numeric score in a new, versioned adapter. Preserve the exact wire value and raw reply. Invalid responses remain invalid.

Exa research found that RunInfra supports schema-constrained JSON through `response_format.type: "json_schema"`. Its cookbook documents `strict: true`, required properties, and `additionalProperties: false`. Plain `json_object` mode does not enforce a specific shape. Model-specific behavior still requires a live conformance check. Sources: [RunInfra structured output](https://runinfra.ai/docs/cookbook/structured-output), [Chat Completions contract](https://runinfra.ai/docs/api-reference/chat-completions), and [the model page](https://runinfra.ai/inference-api/deepseek-v4-1-flash-1017), checked on 26 October 2026.

The earlier 41-image RunInfra run already sent strict JSON Schema with a numeric enum of all 41 allowed tenths. All 24 long decimal literals exactly match a 17-digit floating-point rendering of an allowed enum value. For example, 3.9 was written as `3.8999999999999999`. The long values were present in the saved server response content. This supports a numeric serialization explanation; it does not identify the server component responsible. One of those replies also exceeded the 30-word evidence limit. The prior records remain unchanged.

JSON Schema constrains numeric values rather than their written decimal length. `multipleOf: 0.01` would permit hundredths such as 3.91, changing the scoring scale. It would not distinguish 3.9 from 3.9000. A string enum constrains the exact text and avoids decimal-number serialization. Sources: [numeric semantics](https://json-schema.org/understanding-json-schema/reference/numeric) and [enumerated values](https://json-schema.org/understanding-json-schema/reference/enum).

Generate the score enum without floating-point arithmetic:

```python
score_schema = {
    "type": ["string", "null"],
    "enum": [f"{n // 10}.{n % 10}0" for n in range(10, 51)] + [None],
}
```

Keep the five existing fields: task, status, score, evidence, and limitations. Require every field and reject extra keys. An ok score must use an allowed string; an unscorable result must use null. The evidence remains limited to 30 words. Prompt edits are limited to explaining the string representation and changing the example's score type. The visual rubric, scale anchors, and instruction to avoid invented defects remain unchanged.

This is a new output contract. Decoding a valid string is a declared representation conversion, not a repair of a rejected numeric reply. No old scores or references are converted or relabelled. Five text-only conformance probes must pass before any image in the new trial is sent. All image observations remain pending human review with disabled training masks.
