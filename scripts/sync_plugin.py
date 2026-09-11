# -*- coding: utf-8 -*-
"""Copia el codigo del plugin sobre la instalacion de QGIS, sin reinstalar.

Reinstalar el plugin borra la carpeta entera, y con ella el entorno de Python y los
154 MB de modelos ONNX que viven dentro de trained_models/. Este script copia solo los
ficheros de codigo -los mismos que empaqueta package_plugin.py- y NUNCA toca
trained_models, asi que el entorno sobrevive.

    python scripts/sync_plugin.py                 # busca el perfil automaticamente
    python scripts/sync_plugin.py --perfil palms  # o se le dice cual
    python scripts/sync_plugin.py --destino RUTA  # o la ruta completa del plugin

Despues hay que recargar el plugin en QGIS: el codigo de la ventana esta cacheado en el
proceso de QGIS y no se entera solo. Con el complemento "Plugin Reloader" es un clic;
si no, cerrar y abrir QGIS. Lo que corre en el subproceso -los modelos- se recarga solo.
"""
import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from package_plugin import PLUGIN_NAME, collect_files  # noqa: E402

# Nunca se toca: aqui viven el venv y los modelos.
INTOCABLE = "trained_models"


def destinos_probables():
    """Instalaciones del plugin en los perfiles de QGIS del usuario."""
    base = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "QGIS",
                        "QGIS3", "profiles")
    if not os.path.isdir(base):                      # POSIX
        base = os.path.join(os.path.expanduser("~"), ".local", "share", "QGIS",
                            "QGIS3", "profiles")
    return sorted(glob.glob(os.path.join(base, "*", "python", "plugins",
                                         PLUGIN_NAME)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perfil", help="nombre del perfil de QGIS")
    ap.add_argument("--destino", help="ruta completa de la carpeta del plugin")
    ap.add_argument("-n", "--simular", action="store_true",
                    help="dice que copiaria, sin copiar nada")
    args = ap.parse_args()

    if args.destino:
        destino = args.destino
    else:
        candidatos = destinos_probables()
        if args.perfil:
            candidatos = [c for c in candidatos
                          if os.sep + args.perfil + os.sep in c]
        if not candidatos:
            print("No encuentro el plugin instalado. Usa --destino con la ruta.")
            return 1
        if len(candidatos) > 1:
            print("Hay varias instalaciones; elige una con --perfil o --destino:")
            for c in candidatos:
                print("   " + c)
            return 1
        destino = candidatos[0]

    if not os.path.isdir(destino):
        print("El destino no existe: %s" % destino)
        return 1

    modelos = os.path.join(destino, INTOCABLE)
    n_modelos = len(glob.glob(os.path.join(modelos, "*.onnx")))
    copiados = saltados = 0
    for abs_path, rel_path in collect_files():
        partes = rel_path.replace(os.sep, "/").split("/")
        if partes[0] == INTOCABLE:
            saltados += 1
            continue
        final = os.path.join(destino, rel_path)
        if args.simular:
            copiados += 1
            continue
        os.makedirs(os.path.dirname(final), exist_ok=True)
        shutil.copy2(abs_path, final)
        copiados += 1

    print("%s %d ficheros -> %s"
          % ("copiaria" if args.simular else "copiados", copiados, destino))
    print("   %d saltados por estar en %s/ (venv y modelos intactos)"
          % (saltados, INTOCABLE))
    print("   modelos ONNX presentes en el destino: %d" % n_modelos)
    if not args.simular:
        print("\nRecarga el plugin en QGIS para que la ventana coja los cambios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
