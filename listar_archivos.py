import os

def listar_archivos_y_contenido(ruta_proyecto, archivo_salida):
    """
    Recorre un directorio, ignorando carpetas de librerías,
    y guarda la lista de archivos junto con su contenido
    en un archivo de texto.
    """
    # --- Carpetas que quieres ignorar ---
    carpetas_a_excluir = {
        '__pycache__',
        'venv',
        'env',
        '.venv',
        '.git',  # Común para repositorios de git
        'site-packages'
    }

    # Abre el archivo de salida para escribir. 'w' sobrescribe el archivo si ya existe.
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Análisis del proyecto en: {ruta_proyecto}\n")
        f.write("="*80 + "\n")

        # Recorre el árbol de directorios
        for ruta_actual, directorios, archivos in os.walk(ruta_proyecto):
            
            # Excluye las carpetas no deseadas para que no entre en ellas
            directorios[:] = [d for d in directorios if d not in carpetas_a_excluir]

            # Escribe la ruta de la carpeta actual
            nombre_relativo_carpeta = os.path.relpath(ruta_actual, ruta_proyecto)
            if nombre_relativo_carpeta == '.':
                f.write(f"\n\n📁 Carpeta Raíz: {ruta_proyecto}\n")
            else:
                f.write(f"\n\n📁 Sub-carpeta: {nombre_relativo_carpeta}\n")
            f.write("-" * 50 + "\n")

            # Procesa cada archivo en la carpeta actual
            for archivo in archivos:
                ruta_completa_archivo = os.path.join(ruta_actual, archivo)
                
                # Escribe el nombre del archivo
                f.write(f"\n  📄 Archivo: {archivo}\n")
                f.write(f'{"-"*10} Contenido {"-"*10}\n')

                # --- NUEVA SECCIÓN PARA LEER EL CONTENIDO ---
                try:
                    # Intenta abrir y leer el archivo con codificación UTF-8
                    with open(ruta_completa_archivo, 'r', encoding='utf-8') as contenido_file:
                        contenido = contenido_file.read()
                        f.write(contenido)
                        
                except Exception as e:
                    # Si falla (ej. es un archivo binario o tiene otra codificación),
                    # escribe un mensaje de error en lugar del contenido.
                    f.write(f"\n[No se pudo leer el contenido de este archivo. Razón: {e}]\n")
                
                f.write(f'\n{"-"*10} Fin del Contenido {"-"*10}\n')


# --- CONFIGURACIÓN ---

# 1. Ruta a tu proyecto.
ruta_del_plugin = r'C:\Users\ZEUS\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\deteccion_de_palmeras'

# 2. Nombre del archivo de salida.
nombre_archivo_salida = 'resultado_con_contenido.txt'


# --- EJECUCIÓN ---

if os.path.exists(ruta_del_plugin):
    listar_archivos_y_contenido(ruta_del_plugin, nombre_archivo_salida)
    ruta_absoluta_salida = os.path.abspath(nombre_archivo_salida)
    print(f"✅ ¡Listo! El listado y contenido de los archivos se ha guardado en: {ruta_absoluta_salida}")
else:
    print(f"❌ ¡Error! La ruta del proyecto no existe: {ruta_del_plugin}")