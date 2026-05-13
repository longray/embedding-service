## ADDED Requirements

### Requirement: Batch Entity Creation Endpoint
The system SHALL provide a `POST /api/v1/entities/batch` endpoint that allows creating multiple entities in a single request.

#### Scenario: Successful batch creation
- **WHEN** client sends POST request to `/api/v1/entities/batch` with valid entities array
- **THEN** system creates all valid entities and returns 200 with creation statistics

#### Scenario: Partial success with invalid entries
- **WHEN** client sends batch request with mix of valid and invalid entities
- **THEN** system creates valid entities, skips invalid ones, and returns partial success response

#### Scenario: Duplicate detection
- **WHEN** client sends batch request containing entities with duplicate abstract + type + tenant_id
- **THEN** system creates first occurrence and marks duplicates as skipped

### Requirement: Batch Size Limit
The system SHALL enforce a maximum batch size of 100 entities per request.

#### Scenario: Exceeding batch size limit
- **WHEN** client sends request with more than 100 entities
- **THEN** system returns 400 Bad Request with error message

### Requirement: Response Format
The system SHALL return a standardized response format for batch operations.

#### Scenario: Successful batch response
- **WHEN** batch creation completes
- **THEN** response includes entities array with status per item, plus total/created/skipped/errors counts
