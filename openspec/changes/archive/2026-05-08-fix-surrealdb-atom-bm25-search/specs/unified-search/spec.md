## MODIFIED Requirements

### Requirement: Atom keyword search uses BM25 in SurrealDB fallback
The system SHALL use BM25 @@ operator instead of CONTAINS for Atom keyword search when Meilisearch is unavailable.

#### Scenario: SurrealDB fallback uses BM25 for Atom
- **GIVEN** Meilisearch is unavailable or disabled
- **AND** user performs unified search with scope="atom"
- **WHEN** system executes keyword search
- **THEN** it SHALL use BM25 @@ operator on atom.content and atom.name fields
- **AND** it SHALL use search::score() for relevance ranking

#### Scenario: Atom search returns BM25 scores
- **GIVEN** SurrealDB fallback is used for Atom search
- **WHEN** search results are returned
- **THEN** each atom result SHALL include BM25 relevance score
- **AND** score SHALL be between 0.0 and 1.0

#### Scenario: Atom search supports CJK text via BM25
- **GIVEN** SurrealDB fallback is used for Atom search
- **WHEN** user searches with Chinese keywords like "测试中文"
- **THEN** system SHALL return relevant results using ngram tokenization
- **AND** results SHALL be ordered by BM25 relevance
