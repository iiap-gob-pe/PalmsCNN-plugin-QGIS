# -*- coding: utf-8 -*-
import os, platform
from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QObject, QProcess, pyqtSignal, QTimer, QProcessEnvironment
import hashlib, urllib.request, shutil, zipfile, socket

class _SeqRunner(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(int)
    def __init__(self, parent=None, env_vars=None):
        super().__init__(parent)
        self._cmds=[]; self._idx=-1; self._p=None; self._env=env_vars or {}
    def start(self, commands):
        self._cmds=list(commands); self._idx=-1; self._next()
    def _apply_env(self, p:QProcess):
        if not self._env: return
        env=QProcessEnvironment.systemEnvironment()
        for k,v in self._env.items(): env.insert(k,v)
        p.setProcessEnvironment(env)
    def _next(self):
        if self._p: self._p.deleteLater(); self._p=None
        self._idx+=1
        if self._idx>=len(self._cmds): self.finished.emit(0); return
        argv=self._cmds[self._idx]; self.log.emit("\n$ "+" ".join(argv)+"\n")
        p=QProcess(); self._apply_env(p)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self.log.emit(
            p.readAllStandardOutput().data().decode("utf-8","replace").rstrip("\n")
        ))
        p.finished.connect(lambda rc,_s: self._done(p,rc))
        self._p=p; p.start(argv[0], argv[1:])       
    def _done(self,p,rc):
        self.log.emit(f"[exit code: {rc}]")
        if rc!=0: self.finished.emit(rc)
        else: QTimer.singleShot(10, self._next)

class EnvCore:
    def __init__(self, plugin_name):
        self.plugin_name=plugin_name
        self.profile_root=QgsApplication.qgisSettingsDirPath()
        self.venv_path=os.path.join(self.profile_root,"python","venvs",plugin_name)
        if platform.system().lower().startswith("win"):
            self.venv_python=os.path.join(self.venv_path,"Scripts","python.exe")
        else:
            self.venv_python=os.path.join(self.venv_path,"bin","python")
        os.makedirs(os.path.dirname(self.venv_path), exist_ok=True)

    def venv_exists(self): return os.path.exists(self.venv_python)

    def bridge_osgeo_commands(self):
        """
        Crea un .pth en site-packages del venv apuntando al site-packages de QGIS
        para que 'from osgeo import gdal' funcione dentro del venv.
        Además, prueba la importación.
        """
        s = self.dll_snippet()
        # El .pth irá en el site-packages del venv (detectado en tiempo de ejecución)
        qgis_sp = self.qgis_site_packages().replace("\\", "/")

        create_pth = (
            "import sys,site,os;"
            f"qgis_sp=r'{qgis_sp}';"
            "venv_sp = site.getsitepackages()[0];"
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

        # MUY IMPORTANTE: sin PYTHONHOME (rompe el venv)
        env.pop("PYTHONHOME", None)
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
    
    def ensure_models(self, log_cb=None):
        """
        Verifica/descarga los modelos ONNX después de crear el venv e instalar librerías.
        Si 'log_cb' se pasa (p. ej., self._append del diálogo), lo usa para loguear; si no, usa print.
        """
        log = log_cb or (lambda m: print(m))
        plugin_dir = os.path.dirname(__file__)
        trained_dir = os.path.join(plugin_dir, "trained_models")
        os.makedirs(trained_dir, exist_ok=True)

        # Timeout razonable para descargas
        socket.setdefaulttimeout(60)

        # 🔗 URLs y SHA-256 publicados en Releases (ajústalos cuando cambies versión)
        models = {
            "deeplab": {
                "filename": "model_deeplabv3_segmentation_v1.onnx",
                "url": "https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS/releases/download/v1.0/model_deeplabv3_segmentation_v1.onnx",
                "sha256": "3d384dad78b36adeb4b4b5b4b191e7c2bda5d91c9153948780c7fa0ce31ec9bd",
                "compressed": False,   # True si subes .zip
            },
            "converted": {
                "filename": "model_dwt_instance_segmenetation_v1.onnx",
                "url": "https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS/releases/download/v1.0/model_dwt_instance_segmenetation_v1.onnx",
                "sha256": "e184b3ca942c2a0cc6117b8586342b715d161cf0beaac030122b5c5e6a676fe8",
                "compressed": False,   # True si subes .zip
            },
        }

        def _sha256(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def _download(url: str, dest: str):
            tmp = dest + ".part"
            log(f"⬇️  Downloading model from:\n{url}")
            try:
                with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, dest)
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                finally:
                    pass
                log(f"❌ Error downloading model: {e}")
                raise

        # Recorre y asegura cada modelo
        for key, m in models.items():
            dst = os.path.join(trained_dir, m["filename"])

            # Si ya existe y el hash coincide → OK
            if os.path.exists(dst):
                if _sha256(dst).lower() == m["sha256"].lower():
                    log(f"✅ Model '{m['filename']}' verified.")
                    continue
                else:
                    log(f"⚠️  Hash does not match for {m['filename']}; will be re-downloaded.")
                    try:
                        os.remove(dst)
                    except Exception:
                        pass

            # Descargar (zip o onnx directo)
            if m["compressed"]:
                zip_dst = dst + ".zip"
                _download(m["url"], zip_dst)
                try:
                    with zipfile.ZipFile(zip_dst, "r") as z:
                        z.extractall(trained_dir)
                finally:
                    try:
                        os.remove(zip_dst)
                    except Exception:
                        pass
            else:
                _download(m["url"], dst)

            # Verificación final
            if os.path.exists(dst) and _sha256(dst).lower() == m["sha256"].lower():
                log(f"✅ Model '{m['filename']}' downloaded and verified correctly.")
            else:
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                except Exception:
                    pass
                log(f"❌ SHA-256 verification failed for {m['filename']}")

        log("📦 Model verification completed.")        
    
    def dll_snippet(self):
        bin_dir,py_dir,dlls_dir,_=self._qgis_paths()
        return f"import os; os.add_dll_directory(r'{bin_dir}'); os.add_dll_directory(r'{dlls_dir}')"

    def numpy2_stack_commands(self):
        host = self.find_embedded_python()
        s = self.dll_snippet()
        return [
            [host, "-m", "venv", self.venv_path],
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


 
