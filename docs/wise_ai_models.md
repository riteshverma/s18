# Wise AI: model routing (Gemma vs MedGemma)

## Request tagging

The Wise frontend should send a distinct `source_system` on `POST /runs` so the backend can apply clinical Ollama routing without affecting general S18 traffic.

- **S18 default**: `source_system` omitted or `"s18"` — uses normal `agent` settings (Gemini default; Ollama only where overridden, e.g. `TestAgent`).
- **Wise**: set `source_system` to one of the values in `config/settings.defaults.json` → `wise_ai.source_systems` (default: `wise`, `wise_ai`).

Example body:

```json
{
  "query": "...",
  "source_system": "wise_ai"
}
```

## Ollama: Gemma vs MedGemma (Wise only)

When **all** of the following hold:

1. The run’s `source_system` is listed under `wise_ai.source_systems`, and  
2. The resolved agent provider is **Ollama** (per-agent override or default), and  
3. `wise_ai.use_medgemma` is `true` or `false`,

the server sets the Ollama model name from `wise_ai.ollama_models`:

| `use_medgemma` | Ollama model key used |
|----------------|------------------------|
| `true`         | `medgemma`             |
| `false`        | `gemma`                |

Defaults in `settings.defaults.json`: `gemma` → `gemma3:4b`, `medgemma` → `medgemma:4b`. Adjust to match the tags you install with `ollama pull` (wrong tags return 404 from Ollama).

## Environment overrides (optional)

| Variable | Effect |
|----------|--------|
| `WISE_AI_USE_MEDGEMMA` | `true` / `1` / `yes` / `on` enables MedGemma for Wise Ollama runs |
| `WISE_AI_MEDGEMMA_MODEL` | Overrides `wise_ai.ollama_models.medgemma` |
| `WISE_AI_GEMMA_MODEL` | Overrides `wise_ai.ollama_models.gemma` |

## Non-goals

- **RAG / embedding** models (`get_model` purposes such as `embedding`, `semantic_chunking`) are unchanged; only agent `ModelManager` calls use this Wise routing.
- **Gemini** agents are unchanged by `use_medgemma`; switch agents to Ollama where you want Gemma/MedGemma locally.
