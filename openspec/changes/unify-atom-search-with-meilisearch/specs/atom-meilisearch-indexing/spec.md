## ADDED Requirements

### Requirement: Atom documents are indexed in Meilisearch
The system SHALL index all Atom documents in Meilisearch with appropriate fields for search.

#### Scenario: Atom creation triggers Meilisearch indexing
- **WHEN** an Atom is created via `create_entity` or `create_atom` API
- **THEN** the Atom SHALL be indexed in Meilisearch within 500ms
- **AND** the indexed document SHALL include: id, name, content, type, local_id, entity_id, tenant_id, tags, heading_level

#### Scenario: Atom update triggers Meilisearch reindexing
- **WHEN** an Atom is updated
- **THEN** the Meilisearch document SHALL be updated within 500ms
- **AND** the document SHALL reflect the new field values

#### Scenario: Atom deletion removes Meilisearch document
- **WHEN** an Atom is deleted
- **THEN** the corresponding Meilisearch document SHALL be removed within 500ms

### Requirement: Atom Meilisearch documents have correct structure
The system SHALL build Meilisearch documents for Atoms with fields optimized for search.

#### Scenario: Atom document structure validation
- **WHEN** an Atom is indexed in Meilisearch
- **THEN** the document SHALL have the following fields:
  - `id`: The Atom's RecordID (e.g., "atom:abc123")
  - `surreal_id`: Same as id for consistency with memory documents
  - `name`: Atom name for title search
  - `content`: Atom content for full-text search
  - `content_zh`: Same as content for Chinese tokenization
  - `type`: Atom type (section, chapter, function, etc.)
  - `local_id`: Plugin-local identifier
  - `entity_id`: Parent entity RecordID
  - `tenant_id`: Tenant identifier
  - `tags`: Array of tags
  - `heading_level`: Heading level 1-6 (if applicable)
  - `created_at`: Creation timestamp

### Requirement: Atom documents use consistent ID format
The system SHALL handle Atom RecordID format in Meilisearch.

#### Scenario: Atom ID conversion
- **WHEN** an Atom with id "atom:abc123" is indexed
- **THEN** the Meilisearch document id SHALL be "atom_abc123" (colon replaced with underscore)
- **AND** the original id SHALL be preserved in the `surreal_id` field

#### Scenario: Atom ID retrieval
- **WHEN** searching Meilisearch returns an Atom document
- **THEN** the system SHALL convert "atom_xxx" back to "atom:xxx" for SurrealDB queries
