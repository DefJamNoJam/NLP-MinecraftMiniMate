@echo off
chcp 65001 > nul

echo ===== Starting Minecraft RAG Assistant =====
echo (external RAG server assumed to be running)

:: optional delay
timeout /t 2 /nobreak > nul

:: launch Electron
npm start
