@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Khong tim thay .venv. Tao moi truong bang: python -m venv .venv
  exit /b 1
)

if /I "%~1"=="prepare" goto prepare
if /I "%~1"=="train" goto train
if /I "%~1"=="predict" goto predict
if /I "%~1"=="notebook" goto notebook
if /I "%~1"=="test" goto test
echo Cach dung: .\xg\run-xg.cmd prepare ^| train ^| predict [file.json] ^| notebook ^| test
exit /b 1

:prepare
"%PYTHON_EXE%" "%PROJECT_ROOT%\xg\src\project.py" prepare
exit /b %ERRORLEVEL%

:train
"%PYTHON_EXE%" "%PROJECT_ROOT%\xg\src\project.py" train
exit /b %ERRORLEVEL%

:predict
set "INPUT_FILE=%~2"
if not defined INPUT_FILE set "INPUT_FILE=%PROJECT_ROOT%\xg\examples\shot.json"
"%PYTHON_EXE%" "%PROJECT_ROOT%\xg\src\project.py" predict --input "%INPUT_FILE%"
exit /b %ERRORLEVEL%

:notebook
"%PYTHON_EXE%" -m notebook "%PROJECT_ROOT%\xg\notebooks\learn_xg.ipynb"
exit /b %ERRORLEVEL%

:test
"%PYTHON_EXE%" -m unittest discover -s "%PROJECT_ROOT%\xg\tests" -v
exit /b %ERRORLEVEL%
