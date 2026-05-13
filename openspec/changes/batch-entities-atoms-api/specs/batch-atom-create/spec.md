## ADDED Requirements

### Requirement: Batch Atom Creation Endpoint
The system SHALL provide a `POST /api/v1/atoms/batch` endpoint that allows creating multiple atoms in a single request.

#### Scenario: Successful batch creation
- **WHEN** client sends POST request to `/api/v1/atoms/batch` with valid atoms array
- **THEN** system creates all valid atoms and returns 200 with creation statistics

#### Scenario: Entity existence validation
- **WHEN** client sends batch request with atoms referencing non-existent entity_id
- **THEN** system marks those atoms as error and continues processing others

#### Scenario: Duplicate local_id detection
- **WHEN** client sends batch request containing atoms with duplicate entity_id + local_id
- **THEN** system creates first occurrence and marks duplicates as skipped

### Requirement: Batch Size Limit
The system SHALL enforce a maximum batch size of 100 atoms per request.

#### Scenario: Exceeding batch size limit
- **WHEN** client sends request with more than 100 atoms
- **THEN** system returns 400 Bad Request with error message

### Requirement: Response Format
The system SHALL return a standardized response format for batch operations.

#### Scenario: Successful batch response
- **WHEN** batch creation completes
- **THEN** response includes atoms array with status per item, plus total/created/skipped/errors counts
