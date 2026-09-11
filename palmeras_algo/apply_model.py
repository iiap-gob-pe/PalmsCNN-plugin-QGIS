import numpy as np
import os
from osgeo import gdal
import onnxruntime as rt
from skimage import morphology

# memory_budget vive en la raiz del plugin, un nivel por encima de palmeras_algo.
# Segun como se importe este modulo -como parte del paquete, o suelto con la carpeta
# en sys.path- hay que buscarlo de una forma u otra; el ultimo recurso es anadir la
# raiz del plugin a mano, para que no dependa de quien nos importe.
try:
    from .. import memory_budget
except (ImportError, ValueError):
    try:
        import memory_budget
    except ImportError:
        import sys as _sys
        _raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _raiz not in _sys.path:
            _sys.path.insert(0, _raiz)
        import memory_budget

### Helper Functions ###

# Tipo que ONNX Runtime exige en la entrada. No es una eleccion nuestra: viene
# grabada en el modelo, y alimentarlo con menos precision de la que declara falla.
_ORT_TO_NUMPY = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)': np.float64,
    'tensor(uint8)': np.uint8,
    'tensor(int8)': np.int8,
}


def rint(num):
    return int(round(num))


def window_weight(size, margin):
    """Peso de cada pixel de la ventana al combinarla con sus vecinas.

    Vale 1 en el nucleo y cae suavemente a 0 en el borde, donde la red tuvo menos
    contexto. Asi el solape que ya existe se aprovecha como votacion en vez de
    tirarse: cada pixel recibe la opinion de las dos o tres ventanas que lo cubren,
    pesada por lo lejos que estaba del borde de cada una.
    """
    ramp = np.ones(size, dtype=np.float32)
    if margin > 0:
        t = (np.arange(margin, dtype=np.float32) + 0.5) / margin
        taper = 0.5 - 0.5 * np.cos(np.pi * t)          # 0 -> 1, suave
        ramp[:margin] = taper
        ramp[size - margin:] = taper[::-1]
    return np.outer(ramp, ramp).astype(np.float32)

def scale_image_mean_std(image, nodata_value=0):
    nodata_mask = np.logical_not(np.all(image == nodata_value, axis=2))
    for b in range(image.shape[2]):
        if np.any(nodata_mask):
            mean = np.mean(image[nodata_mask, b])
            std = np.std(image[nodata_mask, b])
            if std != 0:
                image[nodata_mask, b] = (image[nodata_mask, b] - mean) / std
            else:
                image[nodata_mask, b] = image[nodata_mask, b] - mean
    return image

def band_percentiles(dataset, bands=3, nodata_value=0, block_rows=512):
    """Percentiles 1 y 99 de cada banda leyendo por bloques, sin cargar el ráster.

    Dos pasadas: la primera saca el rango de cada banda, la segunda acumula un
    histograma fino sobre ese rango. Invertir la acumulada da el percentil con un
    error de (max-min)/65536, que frente a valores de reflectancia es nada. Es lo
    mismo que devuelve np.percentile sobre la imagen entera, pero con memoria de un
    bloque en vez de la del ráster.
    """
    height, width = dataset.RasterYSize, dataset.RasterXSize
    n = min(bands, dataset.RasterCount)
    lows = [None] * n
    highs = [None] * n

    for b in range(n):
        band = dataset.GetRasterBand(b + 1)
        lo, hi = None, None
        for y in range(0, height, block_rows):
            rows = min(block_rows, height - y)
            chunk = band.ReadAsArray(0, y, width, rows)
            if chunk is None:
                continue
            chunk = chunk.astype(np.float32, copy=False)
            valid = np.isfinite(chunk) & (chunk != nodata_value)
            if valid.any():
                cur_lo = float(chunk[valid].min())
                cur_hi = float(chunk[valid].max())
                lo = cur_lo if lo is None else min(lo, cur_lo)
                hi = cur_hi if hi is None else max(hi, cur_hi)
            del chunk, valid
        lows[b], highs[b] = lo, hi

    BINS = 1 << 16
    resultado = []
    for b in range(n):
        lo, hi = lows[b], highs[b]
        if lo is None or hi is None or hi <= lo:
            resultado.append((0.0, 1.0))
            continue
        band = dataset.GetRasterBand(b + 1)
        hist = np.zeros(BINS, dtype=np.int64)
        for y in range(0, height, block_rows):
            rows = min(block_rows, height - y)
            chunk = band.ReadAsArray(0, y, width, rows)
            if chunk is None:
                continue
            chunk = chunk.astype(np.float32, copy=False)
            valid = np.isfinite(chunk) & (chunk != nodata_value)
            if valid.any():
                idx = ((chunk[valid] - lo) / (hi - lo) * (BINS - 1)).astype(np.int32)
                np.clip(idx, 0, BINS - 1, out=idx)
                hist += np.bincount(idx, minlength=BINS).astype(np.int64)
                del idx
            del chunk, valid
        total = hist.sum()
        if total == 0:
            resultado.append((lo, hi))
            continue
        acumulada = np.cumsum(hist)
        p01 = int(np.searchsorted(acumulada, total * 0.01))
        p99 = int(np.searchsorted(acumulada, total * 0.99))
        del hist, acumulada
        escala = (hi - lo) / float(BINS - 1)
        resultado.append((lo + p01 * escala, lo + p99 * escala))
    return resultado


def normalize_with_percentiles(image, limits, nodata_value=0):
    """La misma normalización de normalize_image_improved, con percentiles dados.

    Separarla permite que el modo por bloques use los percentiles de TODO el ráster
    y no los del bloque, que darían un realce distinto en cada franja.
    """
    nodata_mask = (image[..., 0] == nodata_value)
    for b in range(1, image.shape[2]):
        nodata_mask &= (image[..., b] == nodata_value)
    nodata_mask = np.logical_not(nodata_mask)

    out = image.astype(np.float32, copy=False)
    for b in range(min(out.shape[2], len(limits))):
        p1, p99 = limits[b]
        if not np.any(nodata_mask):
            continue
        if p99 > p1:
            valores = np.clip((out[nodata_mask, b] - p1) / (p99 - p1), 0, 1)
            out[nodata_mask, b] = np.power(valores, 0.8) * 2 - 1
            del valores
        else:
            maximo = np.max(out[nodata_mask, b])
            if maximo != 0:
                out[nodata_mask, b] = out[nodata_mask, b] / maximo * 2 - 1
    del nodata_mask
    return out


def normalize_image_improved(image, nodata_value=0):
    """
    Normalización mejorada basada en percentiles
    """
    # np.all(..., axis=2) materializaria un booleano (H, W, bandas); encadenar las
    # comparaciones da el mismo resultado con un temporal de (H, W).
    nodata_mask = (image[..., 0] == nodata_value)
    for _b in range(1, image.shape[2]):
        nodata_mask &= (image[..., _b] == nodata_value)
    nodata_mask = np.logical_not(nodata_mask)

    # copy=False: load_and_preprocess_tiff_improved ya devuelve float32, asi que el
    # astype duplicaba el raster entero (500 MB en un ortomosaico de 6500 x 6500).
    # Quien llama reasigna el resultado, de modo que trabajar in situ no cambia nada.
    image_float = image.astype(np.float32, copy=False)
    
    for b in range(image_float.shape[2]):
        if np.any(nodata_mask):
            valid_pixels = image_float[nodata_mask, b]
            if len(valid_pixels) > 0:
                # Usar percentiles para evitar outliers
                p1, p99 = np.percentile(valid_pixels, [1, 99])
                if p99 > p1:
                    # Escalar a [0, 1] usando percentiles
                    image_float[nodata_mask, b] = np.clip((valid_pixels - p1) / (p99 - p1), 0, 1)
                    # Aplicar ajuste gamma para mejor contraste
                    image_float[nodata_mask, b] = np.power(image_float[nodata_mask, b], 0.8)
                    # Escalar a [-1, 1] para el modelo
                    image_float[nodata_mask, b] = image_float[nodata_mask, b] * 2 - 1
                else:
                    # Fallback a normalización simple
                    max_val = np.max(valid_pixels)
                    if max_val != 0:
                        image_float[nodata_mask, b] = valid_pixels / max_val * 2 - 1
    return image_float

def save_window_tiff(window, output_path):
    window = window.copy()
    if window.max() > 1:
        window = window / window.max()
    driver = gdal.GetDriverByName('GTiff')
    height, width, bands = window.shape
    out_dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)
    for i in range(bands):
        out_dataset.GetRasterBand(i + 1).WriteArray(window[:, :, i])
    out_dataset.FlushCache()
    out_dataset = None

def load_and_preprocess_tiff_improved(img_path, bands=3):
    """
    Carga y preprocesamiento mejorado de imágenes TIFF
    """
    dataset = gdal.Open(img_path)
    if dataset is None:
        raise ValueError(f"No se pudo abrir la imagen TIFF: {img_path}")
    
    # Detectar tipo de datos
    data_type = dataset.GetRasterBand(1).DataType
    data_type_name = gdal.GetDataTypeName(data_type)
    print(f"Tipo de datos de la imagen: {data_type} ({data_type_name})")
    
    height, width = dataset.RasterYSize, dataset.RasterXSize
    img = np.zeros((height, width, bands), dtype=np.float32)
    
    for i in range(min(bands, dataset.RasterCount)):
        band = dataset.GetRasterBand(i + 1)
        img_data = band.ReadAsArray()
        img[..., i] = img_data
    
    print(f"Valores de la imagen (min, max, mean): {img.min():.2f}, {img.max():.2f}, {img.mean():.2f}")
    
    # Manejo de valores no válidos
    nodata_count = 0
    nodata_val = dataset.GetRasterBand(1).GetNoDataValue()
    if nodata_val is not None:
        nodata_count = np.sum(img == nodata_val)
        # Reemplazar nodata con 0
        for i in range(bands):
            img[img[..., i] == nodata_val, i] = 0
        print(f"Valor nodata original: {nodata_val}")
        print(f"Pixeles con valor nodata: {nodata_count}")
    
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0
    
    return img, dataset, nodata_val

def save_tiff_mask(mask, output_path, reference_dataset):
    """
    MODIFICACIÓN: Guarda la máscara asegurando bordes consistentes en QGIS
    """
    driver = gdal.GetDriverByName('GTiff')
    
    # Obtener toda la información geográfica del dataset de referencia
    geotransform = reference_dataset.GetGeoTransform()
    projection = reference_dataset.GetProjection()
    
    # Crear dataset con las mismas dimensiones que la máscara
    out_dataset = driver.Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    
    # Configurar la misma geotransform y proyección
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    band = out_dataset.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(mask)

    # Configuración adicional para visualización consistente en QGIS
    band.SetMetadataItem('AREA_OR_POINT', 'Area')
    band.FlushCache()

    out_dataset.FlushCache()
    out_dataset = None
    
    print(f"✓ Máscara guardada: {mask.shape[1]}x{mask.shape[0]} pixels")
    print(f"✓ Geotransform aplicado: {geotransform}")

def postprocess_segmentation_mask(mask, min_region_size=20):
    """
    Postprocesamiento para eliminar ruido en la segmentación
    """
    try:
        cleaned_mask = mask.copy()
        
        # Procesar cada clase de palmas
        for class_id in [1, 2, 3]:
            class_mask = mask == class_id
            if np.sum(class_mask) > 0:
                # Eliminar regiones muy pequeñas
                class_mask_cleaned = morphology.remove_small_objects(class_mask, min_size=min_region_size)
                # Eliminar hoyos pequeños
                class_mask_cleaned = morphology.remove_small_holes(class_mask_cleaned, area_threshold=min_region_size)
                
                cleaned_mask[class_mask & ~class_mask_cleaned] = 0
                cleaned_mask[class_mask_cleaned] = class_id
                    
        return cleaned_mask
    except ImportError:
        print("ADVERTENCIA: skimage no disponible, saltando postprocesamiento")
        return mask

def window_centers(size, window_radius, internal_window_radius):
    """Centros de ventana sobre un eje. Misma aritmetica que el camino normal."""
    centers = list(range(window_radius, size - window_radius + 1,
                         internal_window_radius * 2))
    if centers and centers[-1] < size - window_radius:
        centers.append(size - window_radius)
    return centers


def segment_in_blocks(dataset, session, input_name, output_name, model_dtype,
                      window_radius, internal_window_radius, collist, rowlist,
                      batch_size, blend_seams, background_bias, limits,
                      block_rows, bands=3, nodata_value=0):
    """Segmenta por franjas de filas, liberando cada una antes de leer la siguiente.

    Cada franja se lee con un margen de 2 x window_radius arriba y abajo. Ese margen
    no es capricho: una fila del nucleo puede recibir la contribucion de ventanas
    centradas hasta window_radius mas alla, y esas ventanas a su vez leen otro
    window_radius. Sin el margen completo, las filas del borde de cada franja
    recibirian menos votos que en el proceso de una sola pieza y se verian costuras
    horizontales, justo lo que acabamos de quitar.

    Solo la mascara de salida (1 byte por pixel) vive durante todo el proceso.
    """
    height, width = dataset.RasterYSize, dataset.RasterXSize
    output_mask = np.zeros((height, width), dtype=np.uint8)
    win_size = window_radius * 2
    mm = rint(window_radius - internal_window_radius)
    weight = window_weight(win_size, mm)
    core_weight = weight > 0.05
    margin = window_radius * 2
    step = batch_size if batch_size and batch_size > 0 else max(1, len(rowlist))
    window_count = 0
    n_bandas = min(bands, dataset.RasterCount)

    # Los bloques se resuelven antes de empezar para poder decir de cuantos pasos
    # consta el trabajo: sin el total, el progreso no se puede expresar.
    bloques = []
    for y0 in range(0, height, block_rows):
        y1 = min(height, y0 + block_rows)
        r0, r1 = max(0, y0 - margin), min(height, y1 + margin)
        filas = [r for r in rowlist
                 if r + window_radius > y0 and r - window_radius < y1
                 and r - window_radius >= r0 and r + window_radius <= r1]
        if filas:
            bloques.append((y0, y1, r0, r1, filas))
    pasos_total = max(1, len(bloques) * len(collist))
    pasos = 0

    for y0, y1, r0, r1, filas in bloques:
        alto = r1 - r0
        print(f"Bloque filas {y0}-{y1} (leidas {r0}-{r1}, {len(filas)} centros) "
              f"· {alto * width * bands * 4 / 1048576:.0f} MB de imagen")

        block = np.zeros((alto, width, bands), dtype=np.float32)
        for b in range(n_bandas):
            datos = dataset.GetRasterBand(b + 1).ReadAsArray(0, r0, width, alto)
            if datos is not None:
                block[..., b] = datos
            del datos
        block[np.isnan(block)] = 0
        block[np.isinf(block)] = 0
        block = normalize_with_percentiles(block, limits, nodata_value)

        scores = None
        covered = None
        local = np.zeros((alto, width), dtype=np.uint8)

        for col in collist:
            pasos += 1
            print(f"PROGRESO fase=semantica hechas={pasos} de={pasos_total}")
            for inicio in range(0, len(filas), step):
                trozo = filas[inicio:inicio + step]
                batch = np.empty((len(trozo), win_size, win_size, bands),
                                 dtype=model_dtype)
                for k, row in enumerate(trozo):
                    batch[k] = block[row - window_radius - r0:row + window_radius - r0,
                                     col - window_radius:col + window_radius]
                window_count += len(trozo)

                pred = session.run([output_name], {input_name: batch})[0]
                del batch
                if background_bias:
                    pred = pred.copy()
                    pred[..., 0] -= background_bias

                if blend_seams:
                    if scores is None:
                        scores = np.zeros((alto, width, pred.shape[-1]), dtype=np.float32)
                        covered = np.zeros((alto, width), dtype=bool)
                    for k, row in enumerate(trozo):
                        ly = row - window_radius - r0
                        lx = col - window_radius
                        scores[ly:ly + win_size, lx:lx + win_size] += (
                            pred[k] * weight[..., np.newaxis])
                        covered[ly:ly + win_size, lx:lx + win_size] |= core_weight
                else:
                    for k, row in enumerate(trozo):
                        parche = np.argmax(pred[k], axis=-1).astype(np.uint8)
                        if mm > 0:
                            parche = parche[mm:-mm, mm:-mm]
                        ly = row - internal_window_radius - r0
                        local[ly:ly + internal_window_radius * 2,
                              col - internal_window_radius:
                              col + internal_window_radius] = parche
                        del parche
                del pred

        if blend_seams and scores is not None:
            local = np.argmax(scores, axis=-1).astype(np.uint8)
            local[~covered] = 0
            del scores, covered

        local[block[..., 0] == 0] = 0
        output_mask[y0:y1] = local[y0 - r0:y1 - r0]
        del block, local

    return output_mask, window_count


# Semantic segmentation with ONNX
def apply_semantic_segmentation_onnx(input_file_list, output_folder, model_path, window_radius, internal_window_radius, make_tif=True, scaling='normalize',
                                     batch_size=0, blend_seams=True,
                                     background_bias=0.0, low_memory=False,
                                     memory_budget_mb=0.0):
    os.makedirs(output_folder, exist_ok=True)
    
    # Configurar ONNX Runtime con optimizaciones
    providers = ['CPUExecutionProvider']
    session_options = rt.SessionOptions()
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    
    session = rt.InferenceSession(model_path, providers=providers, sess_options=session_options)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Verificación del modelo
    print("=== VERIFICACIÓN MODELO ONNX ===")
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {session.get_outputs()[0].shape}")

    model_dtype = _ORT_TO_NUMPY.get(session.get_inputs()[0].type, np.float32)
    print(f"Input dtype: {session.get_inputs()[0].type} -> numpy {np.dtype(model_dtype).name}")

    # Si el modelo declara un tamano espacial fijo, un window_radius distinto haria
    # fallar session.run() con un error de dimensiones dificil de leer. Se comprueba
    # aqui para poder decir exactamente que valor acepta.
    _in_shape = session.get_inputs()[0].shape
    _spatial = [d for d in _in_shape[1:3] if isinstance(d, int) and d > 0]
    if _spatial and any(d != window_radius * 2 for d in _spatial):
        raise ValueError(
            "El modelo ONNX espera ventanas de {0} x {0} px, pero window_radius={1} "
            "produce ventanas de {2} x {2}. Usa window_radius={3} con este modelo."
            .format(_spatial[0], window_radius, window_radius * 2, _spatial[0] // 2))

    name_saved = None
    for img_path in input_file_list:
        # --- Proyeccion de memoria antes de reservar nada ---------------------
        probe = gdal.Open(img_path)
        if probe is None:
            raise ValueError(f"No se pudo abrir la imagen TIFF: {img_path}")
        p_height, p_width = probe.RasterYSize, probe.RasterXSize
        p_rows = window_centers(p_height, window_radius, internal_window_radius)
        p_cols = window_centers(p_width, window_radius, internal_window_radius)
        por_lote = batch_size if batch_size and batch_size > 0 else max(1, len(p_rows))

        block_rows = 0
        if low_memory:
            lineas, estimado, libre, total = memory_budget.describe(
                p_width, p_height, window_radius, por_lote, blend=blend_seams)
            for linea in lineas:
                print(linea)
            # Con presupuesto explicito manda ese; si no, el 60 % de lo libre, que
            # deja aire para QGIS, el sistema y la etapa de instancias que viene detras.
            presupuesto = memory_budget_mb if memory_budget_mb > 0 else libre * 0.6
            block_rows = memory_budget.plan_rows(
                p_width, p_height, window_radius, por_lote, presupuesto,
                blend=blend_seams)
            if block_rows:
                bloque = memory_budget.estimate_semantic(
                    p_width, p_height, block_rows + window_radius * 4,
                    window_radius, por_lote, blend=blend_seams)
                print(f"No cabe entero en {presupuesto:.0f} MB: se procesara en "
                      f"franjas de {block_rows} filas (~{bloque['total']:.0f} MB cada una)")
            else:
                print("Cabe entero en el presupuesto: no hace falta trocear")
        probe = None

        if block_rows:
            dataset = gdal.Open(img_path)
            print("Percentiles de cada banda leyendo por bloques...")
            limits = band_percentiles(dataset, bands=3)
            for b, (lo, hi) in enumerate(limits):
                print(f"   banda {b + 1}: p1 {lo:.2f} · p99 {hi:.2f}")
            collist = p_cols
            rowlist = p_rows
            print(f"Número de ventanas: {len(rowlist)} filas x {len(collist)} columnas")
            output_mask, window_count = segment_in_blocks(
                dataset, session, input_name, output_name, model_dtype,
                window_radius, internal_window_radius, collist, rowlist,
                batch_size, blend_seams, background_bias, limits, block_rows)
            print(f"Total de ventanas procesadas: {window_count}")

            print("Aplicando postprocesamiento...")
            print(f"Antes postprocesamiento - Clases: {np.unique(output_mask, return_counts=True)}")
            output_mask_processed = postprocess_segmentation_mask(output_mask,
                                                                 min_region_size=20)
            del output_mask
            print(f"Despues postprocesamiento - Clases: "
                  f"{np.unique(output_mask_processed, return_counts=True)}")

            base_name = os.path.basename(img_path).split('.')[0]
            name_saved = f"{base_name}_argmax.tif"
            if make_tif:
                save_tiff_mask(output_mask_processed,
                               os.path.join(output_folder, name_saved), dataset)
            del output_mask_processed
            print(f"Predicción completada para {img_path}")
            dataset = None
            continue

        # Cargar con preprocesamiento mejorado
        img, dataset, original_nodata = load_and_preprocess_tiff_improved(img_path)
        height, width = img.shape[:2]
        print(f"Tamaño de la imagen TIFF: {height}x{width}")

        # Aplicar preprocesamiento según el tipo de escalado
        if scaling == 'mean_std':
            img = scale_image_mean_std(img)
        elif scaling == 'normalize':
            img = normalize_image_improved(img)

        output_mask = np.zeros((height, width), dtype=np.uint8)
        # Acumulador de puntuaciones. Se reserva al ver la primera prediccion, que es
        # cuando se sabe cuantas clases devuelve el modelo. 'covered' anota que pixeles
        # ha tocado alguna ventana: hace falta aparte porque el modelo devuelve logits,
        # que pueden ser negativos, y no se puede deducir la cobertura del signo de la
        # suma.
        scores = None
        covered = None

        collist = list(range(window_radius, width - window_radius + 1, internal_window_radius * 2))
        if collist and collist[-1] < width - window_radius:
            collist.append(width - window_radius)
        rowlist = list(range(window_radius, height - window_radius + 1, internal_window_radius * 2))
        if rowlist and rowlist[-1] < height - window_radius:
            rowlist.append(height - window_radius)
        print(f"Número de ventanas: {len(rowlist)} filas x {len(collist)} columnas")

        window_count = 0
        win_size = window_radius * 2
        bands = img.shape[2]
        # batch_size acota cuantas ventanas van juntas a session.run. Con 0 se envia la
        # columna entera, que es lo que se hacia siempre.
        step = batch_size if batch_size and batch_size > 0 else max(1, len(rowlist))
        window_mb = win_size * win_size * bands * np.dtype(model_dtype).itemsize / 1048576
        print(f"Lote: hasta {step} ventanas x {window_mb:.2f} MB = "
              f"{step * window_mb:.1f} MB por inferencia")

        for col_idx, col in enumerate(collist):
            # Se descartan filas por aritmetica en vez de recortar la ventana para mirar
            # su shape: un slice con inicio negativo no da error, da una ventana corta.
            rows_for_col = [row for row in rowlist
                            if row - window_radius >= 0 and row + window_radius <= height]
            if not rows_for_col:
                continue
            print(f"Procesando {len(rows_for_col)} ventanas en columna {col}")
            print(f"PROGRESO fase=semantica hechas={col_idx + 1} de={len(collist)}")

            for start in range(0, len(rows_for_col), step):
                chunk_rows = rows_for_col[start:start + step]

                # Un unico buffer por lote. Antes se acumulaba la columna entera en una
                # lista y np.stack anadia encima una segunda copia del mismo tamano.
                batch = np.empty((len(chunk_rows), win_size, win_size, bands),
                                 dtype=model_dtype)
                for k, row in enumerate(chunk_rows):
                    batch[k] = img[row - window_radius:row + window_radius,
                                   col - window_radius:col + window_radius]
                window_count += len(chunk_rows)

                # Verificar rango de datos antes de predicción
                if np.abs(batch.min() - (-1.0)) > 0.1 or np.abs(batch.max() - 1.0) > 0.1:
                    print(f"ADVERTENCIA: Rango de ventana inusual - Min: {batch.min():.3f}, Max: {batch.max():.3f}")

                pred = session.run([output_name], {input_name: batch})[0]
                del batch

                if background_bias:
                    # Se le resta ventaja a la clase fondo antes de decidir. Restarlo
                    # aqui, ventana a ventana, hace que al acumular el sesgo quede
                    # pesado igual que las puntuaciones, sin depender de cuantas
                    # ventanas cubran cada pixel.
                    pred = pred.copy()
                    pred[..., 0] -= background_bias

                if blend_seams:
                    if scores is None:
                        n_classes = pred.shape[-1]
                        scores = np.zeros((height, width, n_classes), dtype=np.float32)
                        covered = np.zeros((height, width), dtype=bool)
                        weight = window_weight(win_size,
                                               rint(window_radius - internal_window_radius))
                        # Solo cuenta como cubierto lo que recibe un peso apreciable: el
                        # borde de la ventana, donde el peso se apaga, no debe decidir.
                        core = weight > 0.05
                        print(f"Acumulador de puntuaciones: {height}x{width}x{n_classes} "
                              f"float32 = {scores.nbytes / 1048576:.0f} MB")
                    for i, row in enumerate(chunk_rows):
                        # Se suma la ventana ENTERA, no solo su nucleo: el solape deja de
                        # descartarse y pasa a votar.
                        y0, y1 = row - window_radius, row + window_radius
                        x0, x1 = col - window_radius, col + window_radius
                        scores[y0:y1, x0:x1] += pred[i] * weight[..., np.newaxis]
                        covered[y0:y1, x0:x1] |= core
                else:
                    for i, row in enumerate(chunk_rows):
                        pred_mask = np.argmax(pred[i], axis=-1).astype(np.uint8)
                        if internal_window_radius < window_radius:
                            mm = rint(window_radius - internal_window_radius)
                            pred_mask = pred_mask[mm:-mm, mm:-mm]
                        output_mask[row - internal_window_radius:row + internal_window_radius,
                                    col - internal_window_radius:col + internal_window_radius] = pred_mask
                del pred

        if blend_seams and scores is not None:
            # Un unico argmax sobre la evidencia sumada. Donde dos ventanas discrepaban,
            # antes ganaba la ultima en escribirse; ahora gana la suma de los votos, y
            # las costuras rectas del mosaico desaparecen.
            print("Combinando ventanas solapadas (argmax sobre la suma de puntuaciones)")
            output_mask = np.argmax(scores, axis=-1).astype(np.uint8)
            output_mask[~covered] = 0                     # sin ninguna ventana encima
            print(f"Cubierto por alguna ventana: {covered.mean() * 100:.1f} % del raster")
            del scores, covered

        print(f"Total de ventanas procesadas: {window_count}")
        
        # Aplicar máscara de píxeles válidos
        output_mask[img[..., 0] == 0] = 0
        
        # APLICAR POSTPROCESAMIENTO MEJORADO
        print("Aplicando postprocesamiento...")
        original_stats = np.unique(output_mask, return_counts=True)
        print(f"Antes postprocesamiento - Clases: {original_stats}")
        
        output_mask_processed = postprocess_segmentation_mask(output_mask, min_region_size=20)
        
        final_stats = np.unique(output_mask_processed, return_counts=True)
        print(f"Despues postprocesamiento - Clases: {final_stats}")

        base_name = os.path.basename(img_path).split('.')[0]
        name_saved = f"{base_name}_argmax.tif"
        if make_tif:
            tif_output_path = os.path.join(output_folder, name_saved)
            save_tiff_mask(output_mask_processed, tif_output_path, dataset)

        print(f"Predicción completada para {img_path}")
        dataset = None

    return name_saved