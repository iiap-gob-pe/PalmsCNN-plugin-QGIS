# -*- coding: utf-8 -*-
import os, platform
from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QObject, QProcess, pyqtSignal, QTimer, QProcessEnvironment
import hashlib, urllib.request, urllib.parse, shutil, zipfile, socket, glob

_ALLOWED_URL_SCHEMES = ("https",)
_ALLOWED_URL_HOSTS = ("github.com", "objects.githubusercontent.com")


def _safe_urlopen(url, timeout=60):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"Rejected URL scheme: {parsed.scheme!r}")
    if parsed.hostname not in _ALLOWED_URL_HOSTS:
        raise ValueError(f"Rejected URL host: {parsed.hostname!r}")
    req = urllib.request.Request(url, method="GET")
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310 - scheme/host validated above

class _Cancelled(Exception):
    """El usuario aborto la descarga de modelos."""


class _SeqRunner(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(int)
    def __init__(self, parent=None, env_vars=None):
        super().__init__(parent)
        self._cmds=[]; self._idx=-1; self._p=None; self._env=env_vars or {}
        self._aborted=False; self._active=False
    def start(self, commands):
        self._cmds=list(commands); self._idx=-1
        self._aborted=False; self._active=True
        self._next()
    def _apply_env(self, p:QProcess):
        if not self._env: return
        # Se construye desde cero (NO desde systemEnvironment) para que las variables
        # eliminadas en EnvCore.build_env (PYTHONHOME, PYTHONPATH...) no se hereden
        # del proceso de QGIS: si se heredan, el python del venv arranca con el stdlib
        # de QGIS y 'ensurepip' falla con exit status 1.
        env=QProcessEnvironment()
        for k,v in self._env.items(): env.insert(k,v)
        p.setProcessEnvironment(env)
    def _next(self):
        if self._aborted: return          # cancelado mientras el paso estaba encolado
        if self._p: self._p.deleteLater(); self._p=None
        self._idx+=1
        if self._idx>=len(self._cmds):
            self._active=False; self.finished.emit(0); return
        argv=self._cmds[self._idx]; self.log.emit("\n$ "+" ".join(argv)+"\n")
        p=QProcess(); self._apply_env(p)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self.log.emit(
            p.readAllStandardOutput().data().decode("utf-8","replace").rstrip("\n")
        ))
        p.finished.connect(lambda rc,_s: self._done(p,rc))
        p.errorOccurred.connect(lambda e: self._error(p,e))
        self._p=p; p.start(argv[0], argv[1:])
    def is_running(self):
        # La secuencia sigue viva aunque justo ahora no haya proceso: entre comandos
        # _next() esta encolado en un QTimer.
        return self._active
    def cancel(self):
        """Corta la secuencia y mata el proceso en curso. Se usa kill() y no terminate()
        porque pip en Windows no atiende la peticion suave."""
        if not self._active: return
        self._aborted=True; self._active=False
        p=self._p
        if p is not None and p.state()!=QProcess.NotRunning:
            p.kill(); p.waitForFinished(5000)
        self.log.emit("[cancelled]")
        self.finished.emit(-1)
    def _error(self,p,err):
        # FailedToStart no emite finished(): hay que cortar la secuencia a mano.
        if err==QProcess.FailedToStart and not self._aborted:
            self._aborted=True; self._active=False
            self.log.emit(f"[could not start: {p.program()}]")
            self.finished.emit(1)
    def _done(self,p,rc):
        if self._aborted: return
        self.log.emit(f"[exit code: {rc}]")
        if rc!=0:
            self._aborted=True; self._active=False; self.finished.emit(rc)
        else: QTimer.singleShot(10, self._next)

class EnvCore:
    def __init__(self, plugin_name):
        self.plugin_name=plugin_name
        self.profile_root=QgsApplication.qgisSettingsDirPath()
        #self.venv_path=os.path.join(self.profile_root,"python","venvs",plugin_name)
        plugin_dir = os.path.dirname(__file__)
        self.venv_path=os.path.join(plugin_dir, "trained_models")
        if platform.system().lower().startswith("win"):
            self.venv_python=os.path.join(self.venv_path,"Scripts","python.exe")
        else:
            self.venv_python=os.path.join(self.venv_path,"bin","python")
        os.makedirs(os.path.dirname(self.venv_path), exist_ok=True)

    def venv_site_packages(self):
        """site-packages del venv (Windows y POSIX)."""
        cands = glob.glob(os.path.join(self.venv_path, "Lib", "site-packages"))
        cands += glob.glob(os.path.join(self.venv_path, "lib", "python*", "site-packages"))
        return cands[0] if cands else ""

    def venv_exists(self):
        # No basta con python.exe: un intento fallido deja el interprete copiado pero
        # sin pip ni dependencias. El .pth del puente osgeo es el ultimo paso de la
        # preparacion, asi que marca un entorno realmente completo.
        if not os.path.exists(self.venv_python):
            return False
        sp = self.venv_site_packages()
        return bool(sp) and os.path.exists(os.path.join(sp, "osgeo_qgis.pth"))

    def bridge_osgeo_commands(self):
        """
        Crea un .pth en site-packages del venv apuntando al site-packages de QGIS
        para que 'from osgeo import gdal' funcione dentro del venv.
        Además, prueba la importación.
        """
        s = self.dll_snippet()
        # El .pth irá en el site-packages del venv (detectado en tiempo de ejecución)
        qgis_sp = self.qgis_site_packages().replace("\\", "/")

        # OJO: site.getsitepackages()[0] es la RAIZ del venv en Windows, y ahi los .pth
        # se ignoran (solo se procesan dentro de site-packages). sysconfig purelib da la
        # carpeta correcta en Windows y en POSIX.
        create_pth = (
            "import sys,os,sysconfig;"
            f"qgis_sp=r'{qgis_sp}';"
            "venv_sp = sysconfig.get_paths()['purelib'];"
            "os.makedirs(venv_sp, exist_ok=True);"
            "stale = os.path.join(sys.prefix, 'osgeo_qgis.pth');"
            "os.path.exists(stale) and os.remove(stale);"
            "pth = os.path.join(venv_sp, 'osgeo_qgis.pth');"
            "open(pth,'w',encoding='utf-8').write(qgis_sp + '\\n');"
            "print('Wrote', pth)"
        )
        test_import = "import importlib; importlib.import_module('osgeo.gdal'); print('osgeo.gdal OK')"
        return [
            [ self.venv_python, "-c", s + "; " + create_pth ],
            [ self.venv_python, "-c", s + "; " + test_import ],
        ]

    def find_embedded_python(self):
        base=QgsApplication.prefixPath()
        apps_dir=os.path.abspath(os.path.join(base,".."))
        if os.path.isdir(apps_dir):
            for n in sorted(os.listdir(apps_dir)):
                if n.lower().startswith("python3"):
                    c=os.path.join(apps_dir,n,"python.exe")
                    if os.path.exists(c): return c
        return os.path.join(apps_dir,"Python39","python.exe")

    def _qgis_paths(self):
        apps_dir=os.path.abspath(os.path.join(QgsApplication.prefixPath(),".."))
        bin_dir=os.path.abspath(os.path.join(apps_dir,"..","bin"))
        py_dir=None
        for n in os.listdir(apps_dir):
            if n.lower().startswith("python3"): py_dir=os.path.join(apps_dir,n); break
        if not py_dir: py_dir=os.path.join(apps_dir,"Python39")
        dlls_dir=os.path.join(py_dir,"DLLs"); qt_bin=os.path.join(apps_dir,"Qt5","bin")
        return bin_dir, py_dir, dlls_dir, qt_bin

    def build_env(self):
        bin_dir, py_dir, dlls_dir, qt_bin = self._qgis_paths()
        env = dict(os.environ)
        sep = ";" if os.name == "nt" else ":"
        env["QGIS_PY_SITE"] = os.path.join(py_dir, "Lib", "site-packages")
        # SOLO DLLs necesarias para SSL/Qt. NO metemos ...\apps\Python39 en PATH:
        prepend = [p for p in (bin_dir, dlls_dir, qt_bin) if os.path.isdir(p)]
        env["PATH"] = sep.join(prepend) + sep + env.get("PATH", "")

        # MUY IMPORTANTE: sin PYTHONHOME ni PYTHONPATH. QGIS los define apuntando a su
        # propio Python; heredarlas rompe el venv (ensurepip / pip fallan con exit 1).
        for _v in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                   "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
            env.pop(_v, None)
        env["PYTHONNOUSERSITE"] = "1"

        # Certificados para pip
        cand_cert = os.path.join(py_dir, "Lib", "site-packages", "pip", "_vendor", "certifi", "cacert.pem")
        if os.path.exists(cand_cert):
            env["SSL_CERT_FILE"] = cand_cert
            env["PIP_CERT"] = cand_cert

        # Pip sin build isolation
        env["PIP_NO_BUILD_ISOLATION"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env
    
    def models_dir(self):
        return os.path.join(os.path.dirname(__file__), "trained_models")

    def _model_specs(self):
        """URLs y SHA-256 publicados en Releases (ajustalos cuando cambies version)."""
        base = "https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS/releases/download/v1.0/"
        return {
            "deeplab": {
                "filename": "model_deeplabv3_segmentation_v1.onnx",
                "url": base + "model_deeplabv3_segmentation_v1.onnx",
                "sha256": "3d384dad78b36adeb4b4b5b4b191e7c2bda5d91c9153948780c7fa0ce31ec9bd",  # pragma: allowlist secret
                "compressed": False,   # True si subes .zip
            },
            "converted": {
                "filename": "model_dwt_instance_segmenetation_v1.onnx",
                "url": base + "model_dwt_instance_segmenetation_v1.onnx",
                "sha256": "e184b3ca942c2a0cc6117b8586342b715d161cf0beaac030122b5c5e6a676fe8",  # pragma: allowlist secret
                "compressed": False,   # True si subes .zip
            },
        }

    def models_ready(self):
        """
        Comprobacion barata (solo presencia y tamano) de que estan los .onnx. No calcula
        el SHA-256: eso son cientos de MB y esto se llama desde el hilo de la GUI; la
        verificacion completa la hace ensure_models en su propio hilo.
        """
        trained_dir = self.models_dir()
        for m in self._model_specs().values():
            dst = os.path.join(trained_dir, m["filename"])
            if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                return False
        return True

    def ensure_models(self, log_cb=None, progress_cb=None, cancel_cb=None):
        """
        Verifica/descarga los modelos ONNX despues de crear el venv e instalar librerias.

        Es bloqueante y puede tardar minutos, asi que esta pensada para correr en un hilo
        aparte (ver ModelDownloadWorker): la descarga va por trozos, informa el avance con
        progress_cb(etiqueta, bytes_hechos, bytes_totales) y consulta cancel_cb() entre
        trozos. log_cb recibe los mensajes de texto. Devuelve True si todos los modelos
        quedaron verificados.
        """
        log = log_cb or (lambda m: print(m))
        progress = progress_cb or (lambda label, done, total: None)
        cancelled = cancel_cb or (lambda: False)

        trained_dir = self.models_dir()
        os.makedirs(trained_dir, exist_ok=True)

        # Timeout razonable para descargas
        socket.setdefaulttimeout(60)

        models = self._model_specs()

        CHUNK = 256 * 1024

        def _sha256(path, label):
            h = hashlib.sha256()
            total = os.path.getsize(path)
            done = 0
            with open(path, "rb") as f:
                while True:
                    if cancelled():
                        raise _Cancelled()
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                    done += len(chunk)
                    progress("Verifying " + label, done, total)
            return h.hexdigest()

        def _download(url, dest, label):
            """Descarga a un .part y devuelve el SHA-256 calculado al vuelo."""
            tmp = dest + ".part"
            log(f"⬇️  Downloading model from:\n{url}")
            h = hashlib.sha256()
            try:
                with _safe_urlopen(url) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    progress("Downloading " + label, 0, total)
                    while True:
                        if cancelled():
                            raise _Cancelled()
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        progress("Downloading " + label, done, total)
                os.replace(tmp, dest)
                return h.hexdigest()
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                if isinstance(e, _Cancelled):
                    raise
                log(f"❌ Error downloading model: {e}")
                raise

        all_ok = True
        try:
            for key, m in models.items():
                dst = os.path.join(trained_dir, m["filename"])
                label = m["filename"]

                # Si ya existe y el hash coincide -> OK
                if os.path.exists(dst):
                    if _sha256(dst, label).lower() == m["sha256"].lower():
                        log(f"✅ Model '{label}' verified.")
                        continue
                    log(f"⚠️  Hash does not match for {label}; will be re-downloaded.")
                    try:
                        os.remove(dst)
                    except OSError:
                        pass

                # Descargar (zip o onnx directo)
                if m["compressed"]:
                    zip_dst = dst + ".zip"
                    _download(m["url"], zip_dst, label)
                    try:
                        with zipfile.ZipFile(zip_dst, "r") as z:
                            z.extractall(trained_dir)
                    finally:
                        try:
                            os.remove(zip_dst)
                        except OSError:
                            pass
                    digest = _sha256(dst, label) if os.path.exists(dst) else ""
                else:
                    # El hash se calcula durante la descarga: no hay segunda pasada al fichero.
                    digest = _download(m["url"], dst, label)

                # Verificacion final
                if os.path.exists(dst) and digest.lower() == m["sha256"].lower():
                    log(f"✅ Model '{label}' downloaded and verified correctly.")
                else:
                    all_ok = False
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                    except OSError:
                        pass
                    log(f"❌ SHA-256 verification failed for {label}")
        except _Cancelled:
            log("🛑 Model download cancelled by the user.")
            return False

        log("📦 Model verification completed.")
        return all_ok

    def dll_snippet(self):
        bin_dir,py_dir,dlls_dir,_=self._qgis_paths()
        return f"import os; os.add_dll_directory(r'{bin_dir}'); os.add_dll_directory(r'{dlls_dir}')"

    def numpy2_stack_commands(self):
        host = self.find_embedded_python()
        s = self.dll_snippet()
        return [
            # --without-pip: 'venv' no lanza ensurepip internamente, de modo que un fallo
            # al instalar pip se ve en el paso siguiente con su mensaje real en el log.
            [host, "-m", "venv", "--without-pip", self.venv_path],
            [self.venv_python, "-m", "ensurepip", "--upgrade", "--default-pip"],
            [self.venv_python, "-c",
             s + "; import ssl, sys; "
                 "print('SSL OK:', hasattr(ssl,'OPENSSL_VERSION'), getattr(ssl,'OPENSSL_VERSION', None)); "
                 "sys.exit(0 if hasattr(ssl,'OPENSSL_VERSION') else 2)"],
            #[self.venv_python, "-c",
            # s + "; import runpy, sys; "
            #     "sys.argv=['pip','install','--upgrade','pip','setuptools','wheel']; "
            #     "runpy.run_module('pip', run_name='__main__')"],
            [self.venv_python, "-c",
             s + "; import runpy, sys; "
                 "sys.argv=['pip','install','--only-binary=:all:','numpy>=2,<3','scipy>=1.11','matplotlib>=3.8',"
                 "'networkx>=3.0','pillow>=10','imageio','lazy_loader','packaging']; "
                 "runpy.run_module('pip', run_name='__main__')"],
            [self.venv_python, "-c",
             s + "; import runpy, sys; "
                 "sys.argv=['pip','install','scikit-image>=0.23.2','onnxruntime']; "
                 "runpy.run_module('pip', run_name='__main__')"],
        ]+self.bridge_osgeo_commands()

    def make_seq_runner(self, parent, log_slot):
        r=_SeqRunner(parent=parent, env_vars=self.build_env())
        r.log.connect(log_slot); return r

    def qgis_site_packages(self):
        """Devuelve la carpeta ...\apps\Python39\Lib\site-packages de QGIS."""
        _, py_dir, _, _ = self._qgis_paths()
        return os.path.join(py_dir, "Lib", "site-packages")


 


class ModelDownloadWorker(QObject):
    """
    Ejecuta EnvCore.ensure_models en un hilo aparte para que el dialogo no se congele
    durante la descarga (los .onnx pesan cientos de MB). Se mueve a un QThread con
    moveToThread y se arranca conectando QThread.started a run().
    """
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)   # etiqueta, bytes hechos, bytes totales (0 = desconocido)
    finished = pyqtSignal(bool)

    def __init__(self, env, parent=None):
        super().__init__(parent)
        self._env = env
        self._cancel = False

    def cancel(self):
        """Se llama desde el hilo de la GUI; run() lo consulta entre trozos."""
        self._cancel = True

    def run(self):
        ok = False
        try:
            ok = self._env.ensure_models(
                log_cb=self.log.emit,
                progress_cb=lambda label, done, total: self.progress.emit(label, done, total),
                cancel_cb=lambda: self._cancel,
            )
        except Exception as e:
            self.log.emit(f"❌ Model step reported an error: {e}")
            ok = False
        self.finished.emit(bool(ok))
