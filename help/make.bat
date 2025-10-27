@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === SIEMPRE ubicarse en la carpeta help ===
cd /d "%~dp0"

REM ROOT = carpeta del plugin (padre de help)
set "ROOT=%~dp0..\"
for %%I in ("%ROOT%") do set "ROOT=%%~fI\"

REM Nombre/carpeta del plugin
set "PLUGIN_NAME=deteccion_de_palmeras"
set "PLUGIN_DIR=%ROOT%"

REM build/dist DENTRO de help
set "HELP_DIR=%~dp0"
for %%I in ("%HELP_DIR%") do set "HELP_DIR=%%~fI\"
set "DIST_DIR=%HELP_DIR%dist"
set "BUILD_DIR=%HELP_DIR%build"
set "BUILD_PLUGIN=%BUILD_DIR%\%PLUGIN_NAME%"

REM ===== Ajuste opcional: incluir modelos (0=no; 1=si)
if "%INCLUDE_MODELS%"=="" set "INCLUDE_MODELS=0"

REM ===== Validación base =====
if not exist "%PLUGIN_DIR%metadata.txt" (
  echo [ERROR] No se encontro metadata.txt en: "%PLUGIN_DIR%"
  exit /b 1
)

REM ===== Leer version de metadata.txt =====
set "VERSION="
for /f "usebackq tokens=1* delims==" %%A in (`findstr /r /c:"^version[ ]*=" "%PLUGIN_DIR%metadata.txt"`) do (
  set "VERSION=%%B"
)
for /f "tokens=* delims= " %%A in ("%VERSION%") do set "VERSION=%%A"
set "VERSION=%VERSION:"=%"
if "%VERSION%"=="" (
  echo [WARN] No se encontro 'version=' en metadata.txt. Usando 0.0.0
  set "VERSION=0.0.0"
)

REM ===== Dispatcher simple (sin goto a etiquetas conflictivas) =====
if /I "%~1"=="clean"  call :TASK_CLEAN  & goto :EOF
if /I "%~1"=="package" call :TASK_PACKAGE & goto :EOF
if /I "%~1"=="install" call :TASK_INSTALL & goto :EOF
if /I "%~1"=="check"  (
  echo ROOT         = %ROOT%
  echo PLUGIN_DIR   = %PLUGIN_DIR%
  echo PLUGIN_NAME  = %PLUGIN_NAME%
  echo VERSION      = %VERSION%
  echo BUILD_DIR    = %BUILD_DIR%
  echo DIST_DIR     = %DIST_DIR%
  echo INCLUDE_MODELS = %INCLUDE_MODELS%
  echo.
  echo Contenido de la raiz del plugin:
  dir "%PLUGIN_DIR%" /b
  goto :EOF
)

REM Por defecto: package
call :TASK_PACKAGE
goto :EOF

:TASK_CLEAN
echo [CLEAN] Limpiando...
for /d /r "%PLUGIN_DIR%" %%D in (__pycache__) do rd /s /q "%%D" 2>nul
for /r "%PLUGIN_DIR%" %%F in (*.pyc *.pyo) do del /q "%%F" 2>nul
if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
if exist "%DIST_DIR%" rd /s /q "%DIST_DIR%"
goto :EOF

:TASK_PACKAGE
call :TASK_CLEAN

echo [PACKAGE] Preparando estructura (lista blanca)...
mkdir "%BUILD_PLUGIN%" 2>nul
mkdir "%DIST_DIR%" 2>nul

REM --- 1) Archivos esenciales en la raiz del plugin ---
for %%F in (
  "__init__.py"
  "_env_core.py"
  "deteccion_de_palmeras.py"
  "deteccion_de_palmeras_algorithm.py"
  "deteccion_de_palmeras_provider.py"
  "palmeras_dependency.py"
  "resources_rc.py"
  "metadata.txt"
  "icon.png"
  "logo.png"
  "README.txt"
  "LICENSE"
) do (
  if exist "%PLUGIN_DIR%%%~F" copy /y "%PLUGIN_DIR%%%~F" "%BUILD_PLUGIN%\" >nul
)

REM --- 2) Submódulos necesarios ---
for %%D in (
  "palmeras_algo"
  "palmerasqgis_algo"
) do (
  if exist "%PLUGIN_DIR%%%~D" robocopy "%PLUGIN_DIR%%%~D" "%BUILD_PLUGIN%\%%~D" /E >nul
)

REM --- 3) i18n: SOLO .qm ---
if exist "%PLUGIN_DIR%i18n" (
  mkdir "%BUILD_PLUGIN%\i18n" 2>nul
  robocopy "%PLUGIN_DIR%i18n" "%BUILD_PLUGIN%\i18n" *.qm /S >nul
)

REM --- 4) help: copiar excepto build/dist ---
if exist "%PLUGIN_DIR%help" (
  mkdir "%BUILD_PLUGIN%\help" 2>nul
  robocopy "%PLUGIN_DIR%help" "%BUILD_PLUGIN%\help" /E /XD "build" "dist" /XF "make.bat" "Makefile" >nul
)

REM --- 5) (opcional) modelos ---
if "%INCLUDE_MODELS%"=="1" (
  if exist "%PLUGIN_DIR%trained_models" (
    echo   [INFO] Incluyendo trained_models (INCLUDE_MODELS=1)
    robocopy "%PLUGIN_DIR%trained_models" "%BUILD_PLUGIN%\trained_models" /E >nul
  )
) else (
  echo   [SKIP] trained_models NO incluido (INCLUDE_MODELS=0)
)

REM --- 6) Limpieza extra ---
for /d %%D in (
  "%BUILD_PLUGIN%\tests"
  "%BUILD_PLUGIN%\test"
  "%BUILD_PLUGIN%\.git"
  "%BUILD_PLUGIN%\.idea"
  "%BUILD_PLUGIN%\.vscode"
  "%BUILD_PLUGIN%\notebooks"
  "%BUILD_PLUGIN%\scripts"
  "%BUILD_PLUGIN%\data"
) do (
  if exist "%%~D" rd /s /q "%%~D"
)

REM --- 7) Crear ZIP con carpeta-raiz del plugin ---
if exist "%DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip" del /q "%DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip"
where powershell >nul 2>&1 || (echo [ERROR] PowerShell no disponible.& exit /b 1)

echo [ZIP] %DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip
powershell -NoProfile -Command ^
 "Compress-Archive -Path '%BUILD_PLUGIN%' -DestinationPath '%DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip' -Force"

if exist "%DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip" (
  echo [OK] ZIP generado correctamente.
) else (
  echo [ERROR] No se genero el ZIP.
  exit /b 1
)
goto :EOF

:TASK_INSTALL
set "QGIS_PLUGINS_DIR=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins"
if not exist "%QGIS_PLUGINS_DIR%" (
  echo [ERROR] No existe: %QGIS_PLUGINS_DIR%
  exit /b 1
)
call :TASK_PACKAGE
echo [INSTALL] Instalando en el perfil QGIS...
if exist "%QGIS_PLUGINS_DIR%\%PLUGIN_NAME%" rd /s /q "%QGIS_PLUGINS_DIR%\%PLUGIN_NAME%"
powershell -NoProfile -Command ^
 "Expand-Archive -Path '%DIST_DIR%\%PLUGIN_NAME%-%VERSION%.zip' -DestinationPath '%QGIS_PLUGINS_DIR%\..' -Force"
echo [OK] Instalado. Reinicia QGIS.
goto :EOF
