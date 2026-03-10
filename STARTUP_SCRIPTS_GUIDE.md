# Smart Startup Scripts Guide

## Overview
Optimized Windows batch scripts with intelligent health checking for embedding services.

## Files

### 1. start_all_services.bat (Main Coordinator)
**Purpose**: One-click startup with smart health detection

**Features**:
- Checks if Embedding Service (port 18000) is already running
- Checks if SurrealDB (port 8000) is already running
- Only starts services that are not running
- Waits for services to be ready before completing
- Visual status summary with colors

**Usage**:
```batch
start_all_services.bat
```

**Workflow**:
1. Check Embedding Service health via HTTP GET /health
2. Check SurrealDB health via TCP socket test
3. Display status summary
4. Start only services that are not running
5. Wait for new services to be ready (max 30s)
6. Display final status

---

### 2. start_embedding_service.bat (Modified)
**Purpose**: Start Embedding Service with health check

**New Features**:
- Pre-flight health check
- Reuses existing healthy service
- Port conflict detection with health verification

**Usage**:
```batch
start_embedding_service.bat
```

**Health Check**:
- Endpoint: `http://localhost:18000/health`
- Method: HTTP GET
- Timeout: 3 seconds
- Success: HTTP 200

---

### 3. start_surrealdb.bat (Modified)
**Purpose**: Start SurrealDB with health check

**New Features**:
- Pre-flight connection check
- Reuses existing healthy service
- Port conflict detection with connection test

**Usage**:
```batch
start_surrealdb.bat
```

**Health Check**:
- Endpoint: `ws://localhost:8000/rpc` (WebSocket)
- Method: TCP socket connection test
- Timeout: 3 seconds
- Success: TCP connection accepted

---

### 4. test_health_check.bat (Diagnostic Tool)
**Purpose**: Test health check functionality

**Tests**:
1. Embedding Service HTTP health endpoint
2. SurrealDB TCP socket connection
3. Port occupancy status

**Usage**:
```batch
test_health_check.bat
```

---

## Architecture

```
User
  |
  v
start_all_services.bat (Coordinator)
  |-- Health Check: Embedding (HTTP 18000)
  |-- Health Check: SurrealDB (TCP 8000)
  |-- Decision: Start / Skip / Error
  |
  +-- start_embedding_service.bat
  |     |-- Health Check
  |     |-- Environment Check
  |     |-- Port Check
  |     +-- Start Service
  |
  +-- start_surrealdb.bat
        |-- Health Check
        |-- Environment Check
        |-- Port Check
        +-- Start Service
```

---

## Health Check Implementation

### Embedding Service (PowerShell)
```powershell
Invoke-WebRequest -Uri 'http://localhost:18000/health' 
                  -Method GET 
                  -TimeoutSec 3 
                  -UseBasicParsing
```

### SurrealDB (PowerShell TCP Socket)
```powershell
$c = New-Object System.Net.Sockets.TcpClient
$c.Connect('localhost', 8000)
$c.Close()
```

**Why PowerShell?**
- Built into all Windows versions
- No external dependencies (unlike curl)
- Supports both HTTP and TCP socket checks
- Better error handling

---

## Usage Scenarios

### Scenario 1: Fresh Start
```batch
start_all_services.bat
```
Both services will start.

### Scenario 2: Services Already Running
```batch
start_all_services.bat
```
Script detects healthy services and skips startup.

### Scenario 3: Partial Start (Only SurrealDB Running)
```batch
start_all_services.bat
```
Only starts Embedding Service.

### Scenario 4: Individual Service Start
```batch
start_embedding_service.bat    # Start only Embedding
start_surrealdb.bat            # Start only SurrealDB
```

### Scenario 5: Health Diagnostics
```batch
test_health_check.bat
```
Check current health status without starting services.

---

## Testing

### Test 1: Fresh Start
1. Ensure no services running
2. Run `start_all_services.bat`
3. Verify both services start
4. Check endpoints respond

### Test 2: Idempotency
1. Keep services running from Test 1
2. Run `start_all_services.bat` again
3. Verify message: "Service is already running and healthy"
4. Services should not restart

### Test 3: Port Conflict Handling
1. Start a dummy service on port 18000
2. Run `start_embedding_service.bat`
3. Verify port conflict detection
4. Verify health check on existing service

### Test 4: Individual Start
1. Run `start_surrealdb.bat`
2. Verify only SurrealDB starts
3. Run `start_embedding_service.bat`
4. Verify only Embedding starts

---

## Troubleshooting

### Health Check Fails but Service is Running
- Check firewall settings
- Verify PowerShell execution policy
- Test manually with `test_health_check.bat`

### Port Already in Use
- Use `netstat -ano | findstr ":18000"` to find process
- Use `taskkill /PID <PID> /F` to stop process

### PowerShell Execution Error
- Run: `powershell -ExecutionPolicy Bypass`
- Or modify script to bypass policy

---

## Configuration

Edit script variables to customize:

```batch
:: In start_all_services.bat
set "EMBEDDING_PORT=18000"
set "SURREALDB_PORT=8000"
set "HEALTH_TIMEOUT=5"
set "MAX_WAIT=30"

:: In individual scripts
set "HEALTH_TIMEOUT=3"
```

---

## Notes

- All scripts use UTF-8 encoding (`chcp 65001`)
- Color codes work in Windows 10/11 CMD
- Scripts detect own directory (`%~dp0`)
- Timeout values are in seconds
- All health checks are non-blocking

