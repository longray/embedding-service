## ADDED Requirements

### Requirement: Unified search endpoint returns mixed Entity + Atom results

The system SHALL provide a `POST /api/v1/search` endpoint that accepts a search query and returns results from both Entity (Meilisearch) and Atom (SurrealDB) sources, merged and sorted by relevance score.

#### Scenario: Search returns mixed results

- **WHEN** client sends `POST /api/v1/search` with `{"query": "Vue", "mode": "hybrid"}`
- **THEN** system returns results containing both `type: "entity"` and `type: "atom"` items, sorted by score descending

#### Scenario: Search with scope filter

- **WHEN** client sends `POST /api/v1/search` with `{"query": "test", "scope": "code"}`
- **THEN** system returns only results matching the code scope

#### Scenario: Search with type filters

- **WHEN** client sends `POST /api/v1/search` with `{"query": "setup", "atom_types": ["chapter", "section"]}`
- **THEN** system returns only Atom results whose type is chapter or section

#### Scenario: Search with pagination

- **WHEN** client sends `POST /api/v1/search` with `{"query": "test", "limit": 10}`
- **THEN** system returns at most 10 results with total count

### Requirement: Search request model validation

The system SHALL validate all search request parameters according to the following constraints:

#### Scenario: Invalid mode rejected

- **WHEN** client sends `POST /api/v1/search` with `{"query": "test", "mode": "invalid"}`
- **THEN** system returns 400 error

#### Scenario: Limit bounds enforced

- **WHEN** client sends `POST /api/v1/search` with `{"query": "test", "limit": 200}`
- **THEN** system returns 400 error (max 100)

### Requirement: Entity search uses Meilisearch

The system SHALL search Entity records via Meilisearch full-text index when available, falling back to SurrealDB BM25 when Meilisearch is disabled.

#### Scenario: Meilisearch enabled

- **WHEN** Meilisearch is enabled and client searches
- **THEN** Entity results come from Meilisearch with relevance scores

### Requirement: Atom search uses SurrealDB

The system SHALL search Atom records via SurrealDB query with content LIKE matching and field filters.

#### Scenario: Atom keyword search

- **WHEN** client searches with mode="keyword" and query="setup"
- **THEN** system queries SurrealDB `atom` table with `content LIKE '%setup%'` OR `name LIKE '%setup%'`

### Requirement: Search response format

The system SHALL return a standardized response with results array, total count, mode, and original query.

#### Scenario: Successful search response

- **WHEN** search completes successfully
- **THEN** response contains `{"results": [...], "total": N, "mode": "hybrid", "query": "original query"}`
