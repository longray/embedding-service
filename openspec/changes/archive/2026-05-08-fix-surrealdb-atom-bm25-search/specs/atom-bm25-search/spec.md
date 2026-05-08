## ADDED Requirements

### Requirement: Atom table has BM25 full-text indexes
The system SHALL define BM25 full-text indexes on the atom table for content and name fields.

#### Scenario: Atom content BM25 index exists
- **WHEN** querying SurrealDB schema
- **THEN** idx_atom_content_ft index SHALL exist on atom.content field

#### Scenario: Atom name BM25 index exists
- **WHEN** querying SurrealDB schema
- **THEN** idx_atom_name_ft index SHALL exist on atom.name field

### Requirement: Atom BM25 uses ngram analyzer
The system SHALL use ngram(2,8) analyzer for atom BM25 indexes to support CJK text.

#### Scenario: Atom analyzer configuration
- **WHEN** checking atom_analyzer configuration
- **THEN** it SHALL use TOKENIZERS class and FILTERS lowercase, ngram(2,8)

## MODIFIED Requirements

### Requirement: Atom keyword search uses BM25
The system SHALL use BM25 @@ operator instead of CONTAINS for Atom keyword search in SurrealDB fallback path.

#### Scenario: Meilisearch unavailable - Atom BM25 search
- **GIVEN** Meilisearch is unavailable
- **WHEN** user searches for atoms with keyword "测试"
- **THEN** system SHALL use @@ operator with BM25 scoring
- **AND** return results ordered by relevance score

#### Scenario: Atom search returns BM25 score
- **WHEN** searching atoms via SurrealDB fallback
- **THEN** each result SHALL include BM25 relevance score (not hardcoded 0.5)
