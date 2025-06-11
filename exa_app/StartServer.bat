@echo off
cd %~dp0
call activate app
python src\rag_server.py
pause
