:: 7. Start local web server and launch default browser
echo [STEP 7] Starting local HTTP server on port 8000...
echo The interactive biometric scanner and research dashboard will load.
echo.
echo Launching default web browser pointing to http://localhost:8000/frontend/ ...
start "" "http://localhost:8000/frontend/"

:: Run the local web server
python -m http.server 8000