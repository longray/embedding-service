## 1. Fix Entity Batch Response

- [x] 1.1 Update BatchEntityResponse model to use `entities`, `created`, `skipped`, `errors` fields
- [x] 1.2 Update batch_create_entities to return items with status field
- [x] 1.3 Add duplicate detection for abstract+type+tenant_id
- [x] 1.4 Update response to include all items in `entities` array with status

## 2. Fix Atom Batch Response

- [x] 2.1 Update BatchAtomResponse model to use `atoms`, `created`, `skipped`, `errors` fields
- [x] 2.2 Update batch_create_atoms to return items with status field
- [x] 2.3 Add entity_id existence validation
- [x] 2.4 Add duplicate detection for entity_id+local_id
- [x] 2.5 Update response to include all items in `atoms` array with status

## 3. Testing

- [x] 3.1 Test entity batch with valid items
- [x] 3.2 Test entity batch with duplicates
- [x] 3.3 Test atom batch with valid items
- [x] 3.4 Test atom batch with non-existent entity_id
- [x] 3.5 Test atom batch with duplicate local_id
- [x] 3.6 Verify response format matches spec

## 4. Documentation

- [x] 4.1 Update API documentation
- [x] 4.2 Add response format examples
