@echo off
echo ========================================
echo     Avvio Streamlit App
echo     Video Tracking & Post-Processing
echo ========================================
echo.

REM Controlla se streamlit è installato
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo [!] Streamlit non trovato. Installazione in corso...
    pip install streamlit
    echo.
)

echo [*] Avvio applicazione Streamlit...
echo [*] L'app si aprirà automaticamente nel browser
echo [*] Premi CTRL+C per terminare
echo.

streamlit run app.py

pause
