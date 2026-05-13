## MODIFIED Requirements

### Requirement: Batch Entity Creation Response Format
The system SHALL return response with `entities` array containing all items with their status.

#### Scenario: Successful batch creation response
- **WHEN** batch creation completes
- **THEN** response includes `entities` array with each item having `status: "created"`
- **AND** response includes `created`, `skipped`, `errors` counts

#### Scenario: Partial success response
- **WHEN** some entities fail to create
- **THEN** failed items have `status: "error"` and `error` field
- **AND** response includes accurate `created`, `skipped`, `errors` counts

### Requirement: Entity Duplicate Detection
The system SHALL detect duplicates based on `abstract + type + tenant_id` combination.

#### Scenario: Duplicate entity in request
- **WHEN** request contains entities with same abstract+type+tenant_id
- **THEN** first is created, subsequent are marked `status: "skipped"`
- **AND** skipped items have `error: "duplicate"`

### Requirement: Entity Batch Response Fields
The system SHALL use field names: `entities`, `total`, `created`, `skipped`, `errors`.

#### Scenario: Response structure
- **WHEN** batch endpoint returns
- **THEN** response has `entities` (array), `total` (int), `created` (int), `skipped` (int), `errors` (int)
- **AND** `total` equals `created` + `skipped` + `errors`
