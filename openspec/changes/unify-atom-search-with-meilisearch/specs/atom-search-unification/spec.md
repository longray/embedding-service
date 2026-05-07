## ADDED Requirements

### Requirement: Atom keyword search uses Meilisearch
The system SHALL use Meilisearch for Atom keyword search instead of SurrealDB CONTAINS.

#### Scenario: Atom keyword search via Meilisearch
- **WHEN** a keyword search is performed with scope "atom"
- **THEN** the system SHALL query Meilisearch index
- **AND** the search SHALL support Chinese tokenization
- **AND** the search SHALL support code term dictionary matching
- **AND** results SHALL be ranked by BM25 score

#### Scenario: Atom keyword search with filters
- **WHEN** an Atom keyword search includes filters (type, tags, entity_id)
- **THEN** the system SHALL apply filters in Meilisearch query
- **AND** results SHALL match all filter criteria

#### Scenario: Atom keyword search performance
- **WHEN** an Atom keyword search is performed
- **THEN** response time SHALL be under 100ms for typical queries
- **AND** the search SHALL handle Chinese queries with proper tokenization

### Requirement: Unified search results for Atom and Entity
The system SHALL provide unified search results when searching across both Atom and Entity.

#### Scenario: Hybrid search includes both Atom and Entity
- **WHEN** a hybrid search is performed with scope "all" or "atom"
- **THEN** results SHALL include both Atom and Entity documents
- **AND** results SHALL be ranked using consistent scoring
- **AND** each result SHALL indicate its type (atom or entity)

#### Scenario: Atom search response format
- **WHEN** Atom search returns results
- **THEN** each result SHALL include:
  - `type`: "atom"
  - `id`: Atom RecordID
  - `local_id`: Atom local_id
  - `name`: Atom name
  - `content`: Atom content (truncated if needed)
  - `atom_type`: Atom type
  - `entity_id`: Parent entity ID
  - `score`: Search relevance score
  - `heading_level`: Heading level (if applicable)

### Requirement: Atom search maintains backward compatibility
The system SHALL maintain backward compatibility for Atom search API.

#### Scenario: Existing Atom search API compatibility
- **WHEN** Atom search is performed via existing API endpoints
- **THEN** response format SHALL remain compatible with previous version
- **AND** existing query parameters SHALL continue to work
- **AND** only the search quality SHALL improve (higher recall for Chinese)
