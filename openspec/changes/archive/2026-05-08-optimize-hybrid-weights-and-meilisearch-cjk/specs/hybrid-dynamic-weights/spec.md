## ADDED Requirements

### Requirement: Hybrid search uses dynamic weights based on query language
The system SHALL adjust RRF hybrid weights based on whether the query contains Chinese characters.

#### Scenario: Chinese query uses balanced weights
- **GIVEN** a search query containing Chinese characters like "用户服务"
- **WHEN** hybrid search is performed
- **THEN** vector weight SHALL be 0.5 and keyword weight SHALL be 0.5

#### Scenario: English query uses vector-heavy weights
- **GIVEN** a search query containing only English characters like "user service"
- **WHEN** hybrid search is performed
- **THEN** vector weight SHALL be 0.6 and keyword weight SHALL be 0.4

#### Scenario: Mixed query detected as Chinese
- **GIVEN** a search query containing both Chinese and English like "user 服务"
- **WHEN** hybrid search is performed
- **THEN** weights SHALL be 0.5/0.5 (Chinese mode)

## MODIFIED Requirements

### Requirement: Atom hybrid search applies dynamic weights
The system SHALL use language-aware weights for atom hybrid search instead of hardcoded values.

#### Scenario: Atom hybrid search with Chinese query
- **GIVEN** an atom hybrid search with Chinese query
- **WHEN** RRF fusion is calculated
- **THEN** weights SHALL be determined by query language
- **AND** results SHALL be ranked accordingly
