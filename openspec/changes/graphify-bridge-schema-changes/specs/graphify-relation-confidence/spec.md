## ADDED Requirements

### Requirement: Reference Confidence Fields
The system SHALL support confidence and confidence_score fields on reference relations.

#### Scenario: Create relation with confidence
- **WHEN** client creates relation with confidence="EXTRACTED"
- **THEN** relation is created with confidence field set

#### Scenario: Create relation with confidence_score
- **WHEN** client creates relation with confidence_score=0.8
- **THEN** relation is created with confidence_score field set

#### Scenario: Validate confidence_score range
- **WHEN** client provides confidence_score outside 0.0-1.0
- **THEN** return 400 Bad Request with validation error

### Requirement: Reference Type Extensions
The system SHALL support method and imports_from relation types.

#### Scenario: Create method relation
- **WHEN** client creates relation with type="method"
- **THEN** relation is created successfully

#### Scenario: Create imports_from relation
- **WHEN** client creates relation with type="imports_from"
- **THEN** relation is created successfully

### Requirement: Norm Label Support
The system SHALL support norm_label field on entity and atom.

#### Scenario: Create entity with norm_label
- **WHEN** client creates entity with norm_label="wrapperclient"
- **THEN** entity is created with norm_label field

#### Scenario: Create atom with norm_label
- **WHEN** client creates atom with norm_label="getwebsocketclient"
- **THEN** atom is created with norm_label field
