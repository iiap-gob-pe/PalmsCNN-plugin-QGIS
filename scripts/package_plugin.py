#!/usr/bin/env python3
# coding=utf-8
"""Empaqueta el plugin en dist/deteccion_de_palmeras-<version>.zip.

Solo entra en el zip lo que QGIS necesita en tiempo de ejecucion. Todo lo que
es de desarrollo (scripts .sh/.bat, tests, configuracion de linters, el script
de subida al repositorio) se queda fuera: el escaner de plugins.qgis.org lo
marca como fichero sospechoso o como hallazgo de Bandit/Flake8 aunque nunca se
ejecute dentro de QGIS.

Uso:
    python scripts/package_plugin.py
"""

import configparser
import fnmatch
import os
import zipfile

PLUGIN_NAME = "deteccion_de_palmeras"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios que no se empaquetan (se comparan por ruta relativa, con '/').
EXCLUDED_DIRS = (
    ".git",
    ".claude",
    ".vscode",
    ".idea",
    "dist",
    "build",
    "docs",
    "scripts",
    "test",
    "validacion",   # imagen de prueba, etiquetas a mano y salidas crudas
    "help/build",
    "help/dist",
    "__pycache__",
)

# Ficheros que no se empaquetan (patrones fnmatch sobre la ruta relativa).
EXCLUDED_FILES = (
    ".gitignore",
    "pb_tool.cfg",
    "pylintrc",
    "plugin_upload.py",
    "listar_archivos.py",
    "help/make.bat",
    "help/Makefile",
    "help/DEVELOPER.md",
    "*.pyc",
    "*.pyo",
    "*.zip",
    "*.onnx",
    "*.pt",
    "*.h5",
    ".DS_Store",
    "Thumbs.db",
)


def plugin_version():
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(os.path.join(ROOT, "metadata.txt"), encoding="utf-8")
    return parser.get("general", "version")


def is_excluded_dir(rel_dir):
    rel_dir = rel_dir.replace(os.sep, "/")
    for pattern in EXCLUDED_DIRS:
        if rel_dir == pattern or rel_dir.startswith(pattern + "/"):
            return True
    return False


def is_excluded_file(rel_path):
    rel_path = rel_path.replace(os.sep, "/")
    name = os.path.basename(rel_path)
    for pattern in EXCLUDED_FILES:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def collect_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        rel_dir = "" if rel_dir == "." else rel_dir
        # Poda in-place para no descender en lo excluido.
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not is_excluded_dir(os.path.join(rel_dir, d) if rel_dir else d)
        ]
        for filename in sorted(filenames):
            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename
            if is_excluded_file(rel_path):
                continue
            yield os.path.join(dirpath, filename), rel_path


def main():
    version = plugin_version()
    dist_dir = os.path.join(ROOT, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    zip_path = os.path.join(dist_dir, "%s-%s.zip" % (PLUGIN_NAME, version))

    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel_path in collect_files():
            arcname = "%s/%s" % (PLUGIN_NAME, rel_path.replace(os.sep, "/"))
            zf.write(abs_path, arcname)
            print(arcname)
            count += 1

    size_mb = os.path.getsize(zip_path) / (1024.0 * 1024.0)
    print("\n%d ficheros -> %s (%.1f MB)" % (count, zip_path, size_mb))


if __name__ == "__main__":
    main()
