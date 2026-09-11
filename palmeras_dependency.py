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
from qgis.PyQt.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QHBoxLayout, QMessageBox, QProgressBar
)
from qgis.PyQt.QtCore import Qt, QThread
from ._env_core import EnvCore, ModelDownloadWorker

_env = EnvCore(plugin_name="deteccion_de_palmeras_env")


class DependenciesDialog(QDialog):
    """
    Diálogo modal para crear/verificar el entorno aislado.
    """
    def __init__(self, iface=None, parent=None):
        super().__init__(parent or (iface.mainWindow() if iface else None))
        self.setWindowTitle("PalmsCNN plugin dependencies")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
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

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        v.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Exit from the process...")
        v.addWidget(self.log, 1)

        # Trabajo en curso: secuencia de procesos (venv + pip) e hilo de modelos
        self._runner = None
        self._thread = None
        self._worker = None

        # Estado inicial
        _mismatch = _env.numpy_mismatch()
        if _mismatch:
            self.log.appendPlainText(
                "⚠️ The environment has numpy {} but this QGIS build ships numpy {}. "
                "Its GDAL cannot be loaded across that gap. Click ‘Prepare environment’ "
                "to reinstall the stack with the matching version."
                .format(_mismatch[0], _mismatch[1])
            )
        elif _env.venv_exists() and _env.models_ready():
            self.btn_prep.setEnabled(False)
            self.log.appendPlainText(f"✅ Environment detected: {_env.venv_python}")
        elif _env.venv_exists():
            self.log.appendPlainText(
                "⚠️ Environment detected, but the models are missing. "
                "Click ‘Prepare environment’ to download them."
            )
        else:
            self.log.appendPlainText("⚠️ Environment not found. Click ‘Prepare environment’ to create it.")
        
        # Conexiones
        self.btn_prep.clicked.connect(self._on_prepare)
        self.btn_close.clicked.connect(self._on_close)

        self.setLayout(v)

    def _append(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_prepare(self):
        self.btn_prep.setEnabled(False)

        _mismatch = _env.numpy_mismatch()
        if _mismatch and _env.venv_exists():
            # El venv esta completo pero con el numpy equivocado: se reinstala el
            # stack en su sitio, sin rehacer el entorno entero.
            self._append(
                "Reinstalling the dependency stack with {} to match QGIS."
                .format(_env.numpy_requirement())
            )
            cmds = _env.repair_numpy_commands()
        elif _env.venv_exists():
            # Solo faltan (o fallaron) los modelos: se reintenta ese paso.
            self._start_models()
            return
        else:
            self._append("numpy pinned to {} (QGIS ships numpy {})."
                         .format(_env.numpy_requirement(),
                                 _env.qgis_numpy_version() or "unknown"))
            cmds = _env.dependency_stack_commands()
        self._runner = _env.make_seq_runner(parent=self, log_slot=self._append)

        def finished(rc: int):
            self._runner = None
            if rc == 0:
                self._append("✅ Environment set up correctly.")
                self._start_models()
                return
            self._set_cancel_button(False)
            self.btn_prep.setEnabled(True)
            if rc == -1:                      # cancelado por el usuario
                self._append("🛑 Preparation cancelled. The environment is incomplete; "
                             "press ‘Prepare environment’ to finish it.")
            else:
                self._append(f"❌ Error during preparation (exit {rc}).")
                QMessageBox.warning(self, "Error",
                                    "Could not create the environment. Check the log above.")

        self._runner.finished.connect(finished)
        self._append("⏳ Starting to create the environment…")
        self._set_cancel_button(True)
        self._runner.start(cmds)

    # ---------------- estado ocupado / cancelacion ----------------

    def _set_cancel_button(self, busy: bool):
        self.btn_close.setText("Cancel" if busy else "Close")

    def _runner_running(self) -> bool:
        return self._runner is not None and self._runner.is_running()

    def _busy(self) -> bool:
        return self._runner_running() or self._models_running()

    def _stop_all(self):
        """Cancela lo que esté corriendo y espera a que termine de verdad."""
        if self._runner_running():
            self._append("🛑 Cancelling…")
            self._runner.cancel()
        self._stop_models()

    # ---------------- descarga de modelos en segundo plano ----------------

    def _models_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _start_models(self):
        """
        Lanza la verificación/descarga de los modelos en un QThread. Antes corría en el
        hilo de la GUI y el diálogo quedaba congelado durante toda la descarga.
        """
        self._append("⏳ Verifying/downloading models…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)          # indeterminada hasta conocer el tamaño
        self.progress.setFormat("Starting…")
        self._set_cancel_button(True)

        self._thread = QThread(self)
        self._worker = ModelDownloadWorker(_env)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append)
        self._worker.progress.connect(self._on_model_progress)
        self._worker.finished.connect(self._on_models_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    def _on_model_progress(self, label: str, done: int, total: int):
        if total > 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(done * 100 / total))
            self.progress.setFormat(
                f"{label} — %p% ({done / 1048576:.1f}/{total / 1048576:.1f} MB)"
            )
        else:
            self.progress.setRange(0, 0)      # servidor sin Content-Length
            self.progress.setFormat(label)

    def _on_models_finished(self, ok: bool):
        self.progress.setVisible(False)
        self._set_cancel_button(False)
        if ok:
            self._append("✅ Environment and models ready.")
            self.ok_when_closed = True
        else:
            self._append("⚠️ The models are not ready. You can press "
                         "‘Prepare environment’ to retry the download.")
            self.btn_prep.setEnabled(True)

    def _stop_models(self):
        """Pide la cancelación al worker y espera a que el hilo termine de verdad."""
        if not self._models_running():
            return
        self._append("🛑 Cancelling…")
        if self._worker is not None:
            self._worker.cancel()
        self._thread.quit()
        self._thread.wait(60000)

    def _on_close(self):
        if self._busy():
            self._stop_all()
            return                      # el diálogo sigue abierto para ver el log
        if _env.venv_exists() and _env.models_ready():
            self.ok_when_closed = True
        self.accept()

    def closeEvent(self, event):
        # Cerrar con la X destruiría el QProcess/QThread en marcha: cancelar y esperar.
        if self._busy():
            self._stop_all()
            if self._busy():
                self._append("⚠️ Waiting for the current step to stop…")
                event.ignore()
                return
        super().closeEvent(event)
    
def ensure_dependencies(iface=None) -> bool:
    """
    Verifica y prepara dependencias:
    - Si el entorno ya existe, retorna True (no muestra diálogo)
    - Si no existe, muestra el diálogo modal automáticamente
      hasta que el usuario lo cree o cierre.
    """
    if _env.venv_exists() and _env.models_ready():
        return True

    dlg = DependenciesDialog(iface)
    dlg.exec()

    return dlg.ok_when_closed or _env.venv_exists()

