# ========= PalmsCNN QGIS Plugin Makefile (para ubicarse en help/) =========
# build/ y dist/ quedarán dentro de esta carpeta help/

SHELL := /bin/bash

# Ruta de esta carpeta (help/) y del plugin (padre de help/)
HELP_DIR   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT       := $(abspath $(HELP_DIR)/..)
PLUGIN_DIR := $(ROOT)

# Nombre de carpeta del plugin dentro del ZIP
PLUGIN_NAME := deteccion_de_palmeras

# Dirs de trabajo (dentro de help/)
BUILD_DIR     := $(HELP_DIR)/build
DIST_DIR      := $(HELP_DIR)/dist
BUILD_PLUGIN  := $(BUILD_DIR)/$(PLUGIN_NAME)

# Tomar versión desde metadata.txt (linea 'version=1.2.3')
VERSION := $(shell grep -E '^[[:space:]]*version[[:space:]]*=' "$(PLUGIN_DIR)/metadata.txt" | head -n1 | cut -d= -f2 | tr -d '\r' | xargs)
ifeq ($(strip $(VERSION)),)
  VERSION := 0.0.0
  $(info [WARN] No se encontró 'version=' en metadata.txt. Usando $(VERSION))
endif

# Incluir modelos? (export INCLUDE_MODELS=1 para añadir trained_models/)
INCLUDE_MODELS ?= 0

# Utilidades
RSYNC := rsync -a --prune-empty-dirs
ZIPCMD := zip -rq
UNZIP := unzip -oq

# ========== Targets ==========
.PHONY: help clean package install check

help:
	@echo "Targets:"
	@echo "  make package     -> genera help/dist/$(PLUGIN_NAME)-$(VERSION).zip"
	@echo "  make clean       -> limpia help/build y help/dist + caches .pyc"
	@echo "  make install     -> instala el ZIP en el perfil QGIS por defecto (Linux/macOS)"
	@echo "  make check       -> muestra rutas y versión detectada"
	@echo ""
	@echo "Variables:"
	@echo "  INCLUDE_MODELS=1 -> incluye trained_models/ en el ZIP (por defecto 0)"

clean:
	@echo "[CLEAN] Eliminando artefactos…"
	@find "$(PLUGIN_DIR)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find "$(PLUGIN_DIR)" -name "*.pyc" -type f -delete 2>/dev/null || true
	@rm -rf "$(BUILD_DIR)" "$(DIST_DIR)"

check:
	@echo "HELP_DIR      = $(HELP_DIR)"
	@echo "PLUGIN_DIR    = $(PLUGIN_DIR)"
	@echo "PLUGIN_NAME   = $(PLUGIN_NAME)"
	@echo "BUILD_DIR     = $(BUILD_DIR)"
	@echo "DIST_DIR      = $(DIST_DIR)"
	@echo "VERSION       = $(VERSION)"
	@echo "INCLUDE_MODELS= $(INCLUDE_MODELS)"
	@echo
	@echo "Contenido raíz del plugin:"
	@ls -1 "$(PLUGIN_DIR)" || true

package: clean
	@echo "[PACKAGE] Preparando estructura (lista blanca)…"
	@mkdir -p "$(BUILD_PLUGIN)" "$(DIST_DIR)"

	# 1) Archivos esenciales en la raíz del plugin (si existen)
	@set -e; for f in \
		"__init__.py" \
		"_env_core.py" \
		"deteccion_de_palmeras.py" \
		"deteccion_de_palmeras_algorithm.py" \
		"deteccion_de_palmeras_provider.py" \
		"palmeras_dependency.py" \
		"resources_rc.py" \
		"metadata.txt" \
		"icon.png" \
		"logo.png"; do \
		[ -f "$(PLUGIN_DIR)/$$f" ] && cp -p "$(PLUGIN_DIR)/$$f" "$(BUILD_PLUGIN)/" || true; \
	done

	# 2) Submódulos de código necesarios
	@[ -d "$(PLUGIN_DIR)/palmeras_algo" ]      && $(RSYNC) "$(PLUGIN_DIR)/palmeras_algo/"      "$(BUILD_PLUGIN)/palmeras_algo/"      || true
	@[ -d "$(PLUGIN_DIR)/palmerasqgis_algo" ]  && $(RSYNC) "$(PLUGIN_DIR)/palmerasqgis_algo/"  "$(BUILD_PLUGIN)/palmerasqgis_algo/"  || true

	# 3) i18n: SOLO archivos .qm
	@[ -d "$(PLUGIN_DIR)/i18n" ] && mkdir -p "$(BUILD_PLUGIN)/i18n" && \
		find "$(PLUGIN_DIR)/i18n" -maxdepth 1 -type f -name "*.qm" -exec cp -p {} "$(BUILD_PLUGIN)/i18n/" \; || true

	# 4) help/: copiar pero EXCLUIR help/build y help/dist
	@[ -d "$(PLUGIN_DIR)/help" ] && mkdir -p "$(BUILD_PLUGIN)/help" && \
		$(RSYNC) --exclude 'build' --exclude 'dist' --exclude 'make.bat' --exclude 'Makefile' "$(PLUGIN_DIR)/help/" "$(BUILD_PLUGIN)/help/" || true

	# 5) (Opcional) modelos pre-entrenados
ifneq ($(INCLUDE_MODELS),0)
	@[ -d "$(PLUGIN_DIR)/trained_models" ] && echo "  [INFO] Incluyendo trained_models/" && \
		$(RSYNC) "$(PLUGIN_DIR)/trained_models/" "$(BUILD_PLUGIN)/trained_models/" || true
else
	@echo "  [SKIP] trained_models/ NO incluido (INCLUDE_MODELS=$(INCLUDE_MODELS))"
endif

	# 6) Exclusiones duras por si se coló algo
	@rm -rf \
		"$(BUILD_PLUGIN)/tests" \
		"$(BUILD_PLUGIN)/test" \
		"$(BUILD_PLUGIN)/.git" \
		"$(BUILD_PLUGIN)/.idea" \
		"$(BUILD_PLUGIN)/.vscode" \
		"$(BUILD_PLUGIN)/notebooks" \
		"$(BUILD_PLUGIN)/scripts" \
		"$(BUILD_PLUGIN)/data" 2>/dev/null || true
	@find "$(BUILD_PLUGIN)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find "$(BUILD_PLUGIN)" -name "*.pyc" -type f -delete 2>/dev/null || true

	# 7) Crear ZIP con carpeta-raíz del plugin
	@echo "[ZIP] Creando $(DIST_DIR)/$(PLUGIN_NAME)-$(VERSION).zip"
	@cd "$(BUILD_DIR)" && $(ZIPCMD) "$(DIST_DIR)/$(PLUGIN_NAME)-$(VERSION).zip" "$(PLUGIN_NAME)"
	@echo "[OK] ZIP generado en: $(DIST_DIR)/$(PLUGIN_NAME)-$(VERSION).zip"

install: package
	@echo "[INSTALL] Instalando en el perfil QGIS del usuario…"
	@set -e; \
	if [ -d "$$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins" ]; then \
		PDIR="$$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"; \
	elif [ -d "$$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins" ]; then \
		PDIR="$$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"; \
	else \
		echo "[ERROR] No se encontró el directorio de plugins de QGIS (perfil default)."; \
		echo "Cópialo manualmente desde: $(DIST_DIR)/$(PLUGIN_NAME)-$(VERSION).zip"; \
		exit 1; \
	fi; \
	echo "Destino: $$PDIR"; \
	rm -rf "$$PDIR/$(PLUGIN_NAME)"; \
	$(UNZIP) -d "$$PDIR/.." "$(DIST_DIR)/$(PLUGIN_NAME)-$(VERSION).zip"; \
	echo "[OK] Instalado. Reinicia QGIS."
