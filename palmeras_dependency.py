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
                self.ensure_models()
                self.btn_prep.setEnabled(False)
                self.ok_when_closed = True
            else:
                self._append(f"❌ Error during preparation (exit {rc}).")
                QMessageBox.warning(self, "Error", "Falló la creación del entorno. Revisa el log.")

        runner.finished.connect(finished)
        self._append("⏳ Iniciando creación del entorno…")
        runner.start(cmds)

    def _on_close(self):
        if _env.venv_exists():
            self.ok_when_closed = True
        self.accept()
    
    def ensure_models(self):
        """
        Verifica/descarga los modelos ONNX tras la creación del venv.
        Usa el mismo log del diálogo (self.log).
        """
        trained_dir = os.path.join(self.plugin_dir, "trained_models")
        os.makedirs(trained_dir, exist_ok=True)

        # 🔗 Rellena con tus URLs reales de GitHub Releases y SHA-256
        models = {
                "deeplab": {
                    "filename": "model_deeplabv3_segmentation_v1.onnx",
                    "url": "https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS/releases/download/v1.0/model_deeplabv3_segmentation_v1.onnx",
                    "sha256": "3d384dad78b36adeb4b4b5b4b191e7c2bda5d91c9153948780c7fa0ce31ec9bd",
                    "compressed": False,  # pon True si subes .zip
                },
                "converted": {
                    "filename": "model_dwt_instance_segmenetation_v1.onnx",
                    "url": "https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS/releases/download/v1.0/model_dwt_instance_segmenetation_v1.onnx",
                    "sha256": "e184b3ca942c2a0cc6117b8586342b715d161cf0beaac030122b5c5e6a676fe8",
                    "compressed": False,  # pon True si subes .zip
                },
            }


        def _sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def _download(url, dest):
            tmp = dest + ".part"
            self.log(f"⬇️  Descargando modelo desde:\n{url}")
            try:
                with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, dest)
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                self.log(f"❌ Error al descargar modelo: {e}")
                raise

        # Recorre los modelos configurados
        for key, m in models.items():
            dst = os.path.join(trained_dir, m["filename"])

            # Ya existe y pasa verificación → OK
            if os.path.exists(dst):
                if _sha256(dst).lower() == m["sha256"].lower():
                    self.log(f"✅ Modelo '{m['filename']}' verificado.")
                    continue
                else:
                    self.log(f"⚠️  Hash no coincide para {m['filename']}; se re-descargará.")
                    try:
                        os.remove(dst)
                    except Exception:
                        pass

            # Descargar (zip o onnx directo)
            if m["compressed"]:
                zip_dst = dst + ".zip"
                _download(m["url"], zip_dst)
                with zipfile.ZipFile(zip_dst, "r") as z:
                    z.extractall(trained_dir)
                os.remove(zip_dst)
            else:
                _download(m["url"], dst)

            # Verificar integridad
            if _sha256(dst).lower() != m["sha256"].lower():
                try:
                    os.remove(dst)
                except Exception:
                    pass
                self.log(f"❌ Falló la verificación SHA-256 para {m['filename']}")
            else:
                self.log(f"✅ Modelo '{m['filename']}' descargado y verificado correctamente.")

        self.log("📦 Verificación de modelos completada.")


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
