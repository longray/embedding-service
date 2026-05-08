## MODIFIED Requirements

### Requirement: Meilisearch uses cmn locale for Chinese
The system SHALL use "cmn" (Mandarin Chinese) locale code instead of "zho" for Chinese text processing.

#### Scenario: Index configuration uses cmn locale
- **WHEN** Meilisearch index is configured
- **THEN** localizedAttributes SHALL use "cmn" locale
- **AND** Chinese text SHALL be processed with jieba tokenizer

### Requirement: Meilisearch recognizes Chinese punctuation as separators
The system SHALL treat Chinese punctuation marks as word separators.

#### Scenario: Chinese punctuation separates words
- **GIVEN** text containing "用户、服务和数据"
- **WHEN** indexed by Meilisearch
- **THEN** the text SHALL be tokenized as separate words
- **AND** searching for "用户" SHALL match the document

#### Scenario: Chinese punctuation list configured
- **WHEN** Meilisearch settings are applied
- **THEN** separatorTokens SHALL include "、", "；", "："
- **AND" optionally include "，", "。", "？", "！"

## ADDED Requirements

### Requirement: Meilisearch configuration supports CJK text
The system SHALL configure Meilisearch for optimal Chinese, Japanese, and Korean text search.

#### Scenario: CJK text is properly tokenized
- **GIVEN** documents containing CJK text
- **WHEN** indexed and searched
- **THEN** results SHALL be relevant and accurately ranked
