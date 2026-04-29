## MODIFIED Requirements

### Requirement: Atom CRUD supports v3.3 hierarchy fields

The system SHALL accept and store 6 new optional fields in Atom CRUD operations: tags (list[str]), heading_level (int 1-6), parent_id (str|None), order (str), aliases (list[str]), entity_id (str).

#### Scenario: Create Atom with all new fields

- **WHEN** client sends `POST /api/v1/atoms` with `{"type": "chapter", "content": "...", "tags": ["vue"], "heading_level": 1, "parent_id": null, "order": "a0", "aliases": ["Setup API"], "entity_id": "entity:xxx"}`
- **THEN** system creates Atom with all fields stored in SurrealDB

#### Scenario: Create Atom without new fields (backward compatible)

- **WHEN** client sends `POST /api/v1/atoms` with only existing fields
- **THEN** system creates Atom with new fields defaulting to None/empty

#### Scenario: Update Atom with new fields

- **WHEN** client sends `PUT /api/v1/atoms/{id}` with `{"tags": ["updated"], "order": "aV"}`
- **THEN** system updates only the specified new fields

#### Scenario: heading_level validation

- **WHEN** client sends heading_level outside 1-6 range
- **THEN** system returns 400 validation error

### Requirement: Atom types include knowledge document types

The system SHALL accept chapter and section as valid Atom types in addition to existing code analysis types.

#### Scenario: Create chapter Atom

- **WHEN** client sends `POST /api/v1/atoms` with `{"type": "chapter", "content": "..."}`
- **THEN** system creates Atom successfully

#### Scenario: Create section Atom

- **WHEN** client sends `POST /api/v1/atoms` with `{"type": "section", "content": "..."}`
- **THEN** system creates Atom successfully

### Requirement: Atom response includes all new fields

The system SHALL include all 6 new fields in Atom GET/POST/PUT responses, with None/default values for missing data.

#### Scenario: Get Atom returns new fields

- **WHEN** client sends `GET /api/v1/atoms/{id}`
- **THEN** response includes tags, heading_level, parent_id, order, aliases, entity_id fields
