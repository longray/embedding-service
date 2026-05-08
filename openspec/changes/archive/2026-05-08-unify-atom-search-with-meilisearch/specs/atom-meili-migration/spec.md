## ADDED Requirements

### Requirement: Existing Atom data migration to Meilisearch
The system SHALL provide a migration script to index existing Atom data in Meilisearch.

#### Scenario: Migration script execution
- **WHEN** the migration script is executed
- **THEN** all existing Atom documents in SurrealDB SHALL be indexed in Meilisearch
- **AND** the migration SHALL be idempotent (safe to run multiple times)
- **AND** the migration SHALL report progress and statistics

#### Scenario: Migration handles large datasets
- **WHEN** migrating a large number of Atoms (>10,000)
- **THEN** the migration SHALL use batching (batch size 100-500)
- **AND** the migration SHALL handle errors gracefully
- **AND** failed batches SHALL be logged and skippable

#### Scenario: Migration validation
- **WHEN** migration completes
- **THEN** the system SHALL verify Meilisearch document count matches SurrealDB
- **AND** sample documents SHALL be validated for correct field mapping

### Requirement: Migration script features
The system SHALL provide a comprehensive migration script with appropriate options.

#### Scenario: Migration with dry-run option
- **WHEN** migration is run with `--dry-run` flag
- **THEN** the system SHALL report what would be migrated without making changes
- **AND** the report SHALL include: total atoms, estimated time, index requirements

#### Scenario: Migration with batch size option
- **WHEN** migration is run with `--batch-size N` flag
- **THEN** the system SHALL use N as the batch size for indexing

#### Scenario: Migration with tenant filter
- **WHEN** migration is run with `--tenant-id <id>` flag
- **THEN** only Atoms for that tenant SHALL be migrated

#### Scenario: Migration resumption
- **WHEN** migration is interrupted and restarted
- **THEN** the system SHALL skip already-indexed Atoms (based on id)
- **AND** the migration SHALL continue from where it left off
