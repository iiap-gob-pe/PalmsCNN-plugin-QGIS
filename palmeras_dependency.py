# -*- coding: utf-8 -*-
"""
palmeras_dependency.py
----------------------
Versión revisada: el diálogo modal se muestra automáticamente
SIEMPRE que no exista el entorno virtual (venv), sin depender
de la consola ni de valores guardados en QSettings.

- Modal (bloqueante)
- Solo se cierra cuando el usuario termina o cancela
- Logs en vivo con EnvCore
"""
import os, hashlib, urllib.request, shutil, zipfile

from qgis.PyQt.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QHBoxLayout, QMessageBox
)
from qgis.PyQt.QtCore import Qt
from ._env_core import EnvCore

_env = EnvCore(plugin_name="deteccion_de_palmeras_env")


class DependenciesDialog(QDialog):
    """
    Diálogo modal para crear/verificar el entorno aislado.
    """
    def __init__(self, iface=None, parent=None):
        super().__init__(parent or (iface.mainWindow() if iface else None))
        self.setWindowTitle("PalmsCNN plugin dependencies")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(780, 520)
        self.ok_when_closed = False

        w = QWidget(self)
        v = QVBoxLayout(w)

        v.addWidget(QLabel("<b>Prepare virtual environment (venv) for the plugin</b>"))
        info = QLabel(
            "An isolated virtual environment will be created with the necessary dependencies. "
            "to run the palm tree detection model (NumPy 2, scikit-image, onnxruntime)."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        self.btn_prep = QPushButton("Prepare environment")
        self.btn_close = QPushButton("Close")
        hb = QHBoxLayout()
        hb.addWidget(self.btn_prep)
        hb.addWidget(self.btn_close)
        v.addLayout(hb)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Exit from the process...")
        v.addWidget(self.log, 1)

        # Estado inicial
        if _env.venv_exists():
            self.btn_prep.setEnabled(False)
            self.log.appendPlainText(f"✅ Environment detected: {_env.venv_python}")
        else:
            self.log.appendPlainText("⚠️ Environment not found. Click ‘Prepare environment’ to create it.")
        
        # Conexiones
        self.btn_prep.clicked.connect(self._on_prepare)
        self.btn_close.clicked.connect(self._on_close)

        self.setLayout(v)

    def _append(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_prepare(self):
        if _env.venv_exists():
            QMessageBox.information(self, "Dependencies","The environment already exists.")
            self.btn_prep.setEnabled(False)
            return

        cmds = _env.numpy2_stack_commands()
        runner = _env.make_seq_runner(parent=self, log_slot=self._append)

        def finished(rc: int):
            if rc == 0:
                self._append("✅ Environment set up correctly.")
                # ↓↓↓ AHORA descarga/verifica modelos desde EnvCore, reutilizando el log del diálogo
                try:
                    _env.ensure_models(self._append)
                except Exception as e:
                    self._append(f"⚠️ Model step reported an error: {e}")
                self.btn_prep.setEnabled(False)
                self.ok_when_closed = True
            else:
                self._append(f"❌ Error during preparation (exit {rc}).")
                QMessageBox.warning(self, "Error", "Falló la creación del entorno. Revisa el log.")

        runner.finished.connect(finished)
        self._append("⏳ Starting to create the environment…")
        runner.start(cmds)

    def _on_close(self):
        if _env.venv_exists():
            self.ok_when_closed = True
        self.accept()
    
def ensure_dependencies(iface=None) -> bool:
    """
    Verifica y prepara dependencias:
    - Si el entorno ya existe, retorna True (no muestra diálogo)
    - Si no existe, muestra el diálogo modal automáticamente
      hasta que el usuario lo cree o cierre.
    """
    if _env.venv_exists():
        return True

    dlg = DependenciesDialog(iface)
    dlg.exec_()

    return dlg.ok_when_closed or _env.venv_exists()

