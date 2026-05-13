## MODIFIED Requirements

### Requirement: Batch Atom Creation Response Format
The system SHALL return response with `atoms` array containing all items with their status.

#### Scenario: Successful batch creation response
- **WHEN** batch creation completes
- **THEN** response includes `atoms` array with each item having `status: "created"`
- **AND** response includes `created`, `skipped`, `errors` counts

#### Scenario: Partial success response
- **WHEN** some atoms fail to create
- **THEN** failed items have `status: "error"` and `error` field
- **AND** response includes accurate `created`, `skipped`, `errors` counts

### Requirement: Atom Duplicate Detection
The system SHALL detect duplicates based on `entity_id + local_id` combination.

#### Scenario: Duplicate atom in request
- **WHEN** request contains atoms with same entity_id+local_id
- **THEN** first is created, subsequent are marked `status: "skipped"`
- **AND** skipped items have `error: "duplicate local_id"`

### Requirement: Atom Entity ID Validation
The system SHALL validate that `entity_id` exists before creating atom.

#### Scenario: Non-existent entity_id
- **WHEN** atom has entity_id that doesn't exist
- **THEN** atom is marked `status: "error"` with `error: "entity_id not found"`
- **AND** other atoms continue processing

### Requirement: Atom Batch Response Fields
The system SHALL use field names: `atoms`, `total`, `created`, `skipped`, `errors`.

#### Scenario: Response structure
- **WHEN** batch endpoint returns
- **THEN** response has `atoms` (array), `total` (int), `created` (int), `skipped` (int), `errors` (int)
- **AND** `total` equals `created` + `skipped` + `errors`
