## 1. Entity Batch API

- [x] 1.1 Add Pydantic models for batch entity request/response
- [x] 1.2 Implement POST /api/v1/entities/batch endpoint in entity.py
- [x] 1.3 Add duplicate detection logic (abstract + type + tenant_id)
- [x] 1.4 Add batch size limit validation (max 100)

## 2. Atom Batch API

- [x] 2.1 Add Pydantic models for batch atom request/response
- [x] 2.2 Implement POST /api/v1/atoms/batch endpoint in atom.py
- [x] 2.3 Add entity_id existence validation
- [x] 2.4 Add duplicate detection logic (entity_id + local_id)
- [x] 2.5 Add batch size limit validation (max 100)

## 3. Testing & Verification

- [x] 3.1 Test entities/batch endpoint with valid batch
- [x] 3.2 Test entities/batch with duplicates
- [x] 3.3 Test atoms/batch endpoint with valid batch
- [x] 3.4 Test atoms/batch with non-existent entity_id
- [x] 3.5 Verify batch size limit enforcement
- [x] 3.6 Restart service and verify endpoints available

## 4. Documentation

- [x] 4.1 Update API documentation with new endpoints
- [x] 4.2 Add usage examples
- [x] 4.3 Reply to backend-team with completion notice
