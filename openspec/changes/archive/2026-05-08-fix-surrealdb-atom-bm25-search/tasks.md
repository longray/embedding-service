## 1. Schema Changes

- [x] 1.1 Add atom_analyzer definition to init_surrealdb.surql
- [x] 1.2 Add idx_atom_content_ft BM25 index to init_surrealdb.surql
- [x] 1.3 Add idx_atom_name_ft BM25 index to init_surrealdb.surql

## 2. Search Logic Changes

- [x] 2.1 Import _sanitize_query function in search.py
- [x] 2.2 Replace CONTAINS with @@ operator in _search_atoms_by_keyword
- [x] 2.3 Add search::score to SELECT clause
- [x] 2.4 Update result processing to use dynamic BM25 score

## 3. Testing

- [x] 3.1 Run existing search tests to ensure no regression
- [x] 3.2 Run atom-related tests to verify functionality
- [x] 3.3 Verify schema changes with init_database.py --verify-only

## 4. Documentation

- [x] 4.1 Update CHANGELOG.md with fix description
- [x] 4.2 Update AGENTS.md if needed

## 5. Verification

- [x] 5.1 All tests pass
- [x] 5.2 Code review completed
- [x] 5.3 OpenSpec change archived
