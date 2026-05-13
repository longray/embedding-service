## 1. SurrealDB Schema Updates

- [x] 1.1 Add confidence field to reference table
- [x] 1.2 Add confidence_score field to reference table
- [x] 1.3 Update reference.type enum to include method and imports_from
- [x] 1.4 Add norm_label field to entity table
- [x] 1.5 Add norm_label field to atom table

## 2. Pydantic Model Updates

- [x] 2.1 Add METHOD and IMPORTS_FROM to ReferenceType enum
- [x] 2.2 Add confidence and confidence_score to ReferenceCreateRequest
- [x] 2.3 Add confidence and confidence_score to ReferenceResponse
- [x] 2.4 Add norm_label to EntityCreateRequest
- [x] 2.5 Add norm_label to AtomCreateRequest

## 3. API Implementation

- [x] 3.1 Update create_reference to handle new fields
- [x] 3.2 Update create_entity to handle norm_label
- [x] 3.3 Update create_atom to handle norm_label

## 4. Testing

- [x] 4.1 Test creating relation with confidence fields
- [x] 4.2 Test creating method and imports_from relations
- [x] 4.3 Test creating entity with norm_label
- [x] 4.4 Test creating atom with norm_label
- [x] 4.5 Verify backward compatibility

## 5. Documentation

- [x] 5.1 Update API documentation
- [x] 5.2 Add migration notes
