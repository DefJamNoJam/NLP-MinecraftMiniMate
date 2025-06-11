@echo off
echo ===== SLM Server Clean Start =====
echo.
echo 1. Terminating processes using port 8504...

:: 간소화된 방식 사용
for /f "tokens=5" %%a in ('netstat -ano | findstr ":8504" | findstr "LISTENING"') do (
  echo Terminating process (PID: %%a)...
  taskkill /f /pid %%a
  echo Process terminated
)

echo 2. Cleaning up Python processes...
taskkill /f /im python.exe 2>nul
taskkill /f /im pythonw.exe 2>nul

echo 3. Waiting for resources to be released (3 seconds)...
timeout /t 3 /nobreak > nul

echo 4. Starting application...
call start.bat
echo.
echo Application started successfully.
