# Claude API Best Practices for Habitat Evolution

## Overview

This guide provides best practices for using the Anthropic Claude API within the Habitat Evolution system. Following these guidelines will help optimize pattern extraction quality, minimize token usage, and ensure consistent results across the system.

## API Key Management

1. **Environment Variables**: Always store the API key in the `ANTHROPIC_API_KEY` environment variable rather than hardcoding it in source files.

2. **Testing Mode**: When developing or testing, the ClaudeAdapter will automatically fall back to mock mode if no API key is available, allowing development to proceed without API costs.

3. **CI/CD Integration**: For CI/CD pipelines, consider using mock mode by default and only enabling the real API for specific integration tests.

## Prompt Engineering

### General Principles

1. **Be Specific**: Claude performs best when given specific, clear instructions. Avoid ambiguity in your prompts.

2. **Structured Output**: Always request structured output (JSON) when extracting patterns to ensure consistent parsing.

3. **Few-Shot Examples**: For complex pattern extraction, include examples of the patterns you want to extract.

4. **Context Matters**: Provide relevant context to help Claude understand the domain and generate more accurate patterns.

### Pattern Extraction Prompts

When extracting patterns from climate risk documents, follow this structure:

```prompt
You are an assistant that helps extract patterns from climate risk documents.

A pattern is a recurring structure, theme, or concept that appears in the document.
Patterns should be specific, well-defined, and clearly present in the document.

For each pattern you identify, provide:
1. A short name or title
2. A brief description
3. Evidence from the document
4. A quality assessment (hypothetical, emergent, or stable)

Format your response as JSON with the following structure:
{
    "patterns": [
        {
            "name": "Pattern Name",
            "description": "Pattern Description",
            "evidence": "Evidence from document",
            "quality_state": "emergent"
        },
        ...
    ]
}

Ensure your response is valid JSON that can be parsed programmatically.
```

### Query Enhancement Prompts

When enhancing queries with pattern awareness, follow this structure:

```prompt
You are an assistant that helps answer questions based on provided patterns and context.

The following patterns have been identified as relevant to the query:
[Formatted patterns here]

Use these patterns to inform your response, but don't explicitly mention them unless asked.
Your response should be helpful, accurate, and concise.
```

## Model Selection

1. **Default Model**: Use `claude-3-opus-20240229` for most pattern extraction and query enhancement tasks, as it provides the highest quality results.

2. **Lightweight Alternative**: For simpler tasks or when response speed is critical, consider using `claude-3-haiku-20240307` which is faster but may produce lower quality results.

3. **Model Parameters**:
   - Use `max_tokens=1000` for query responses
   - Use `max_tokens=4000` for document pattern extraction
   - Use `temperature=0.2` for more deterministic outputs
   - Use `temperature=0.7` for more creative pattern identification

## Token Optimization

1. **Document Chunking**: For large documents, split them into chunks of approximately 10,000 tokens to avoid hitting token limits.

2. **Metadata Filtering**: Only include relevant metadata in your prompts to reduce token usage.

3. **Pattern Filtering**: When passing patterns to Claude for query enhancement, only include the most relevant patterns (typically 3-5) rather than all available patterns.

4. **Response Length Control**: Use the `max_tokens` parameter to limit response length and control costs.

## Error Handling

1. **Graceful Degradation**: The ClaudeAdapter includes fallback to mock responses if the API call fails, ensuring system stability.

2. **Retry Logic**: For critical operations, implement retry logic with exponential backoff.

3. **JSON Parsing**: Always handle JSON parsing errors gracefully, as Claude may occasionally produce malformed JSON.

4. **Rate Limiting**: Be prepared to handle rate limiting errors from the API by implementing appropriate backoff strategies.

## Metrics and Monitoring

1. **Track Usage**: Use the `claude_metrics` system to track API usage, response times, and token consumption.

2. **Analyze Costs**: Regularly run the metrics analysis script to monitor API costs and optimize usage.

3. **Quality Assessment**: Track pattern extraction quality over time to ensure the system is producing valuable results.

4. **Performance Monitoring**: Monitor response times to identify potential bottlenecks or API performance issues.

## Example Patterns

Here are examples of high-quality patterns extracted from climate risk documents:

### Sea Level Rise Pattern

```json
{
  "name": "Sea Level Rise Impact on Coastal Infrastructure",
  "description": "Pattern of increasing risk to coastal infrastructure due to projected sea level rise, requiring adaptation strategies",
  "evidence": "Martha's Vineyard is projected to experience sea level rise of 1.5 to 3.1 feet by 2070. This will impact coastal properties, infrastructure, and ecosystems.",
  "quality_state": "emergent"
}
```

### Extreme Weather Pattern

```json
{
  "name": "Increasing Extreme Weather Frequency",
  "description": "Pattern of increasing frequency and intensity of extreme weather events due to climate change",
  "evidence": "Climate models project an increase in the frequency and intensity of storms, including more frequent nor'easters with higher storm surge, potential increase in hurricane intensity.",
  "quality_state": "emergent"
}
```

## Conclusion

Following these best practices will help ensure effective use of the Claude API within the Habitat Evolution system, optimizing for both cost and quality while maintaining system stability and performance.
