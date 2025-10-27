import numpy as np
import os
from osgeo import gdal
import onnxruntime as rt
from skimage import morphology

### Helper Functions ###

def rint(num):
    return int(round(num))

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

def normalize_image_improved(image, nodata_value=0):
    """
    Normalización mejorada basada en percentiles
    """
    nodata_mask = np.logical_not(np.all(image == nodata_value, axis=2))
    image_float = image.astype(np.float32)
    
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

# Semantic segmentation with ONNX
def apply_semantic_segmentation_onnx(input_file_list, output_folder, model_path, window_radius, internal_window_radius, make_tif=True, scaling='normalize'):
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

    name_saved = None
    for img_path in input_file_list:
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

        collist = list(range(window_radius, width - window_radius + 1, internal_window_radius * 2))
        if collist and collist[-1] < width - window_radius:
            collist.append(width - window_radius)
        rowlist = list(range(window_radius, height - window_radius + 1, internal_window_radius * 2))
        if rowlist and rowlist[-1] < height - window_radius:
            rowlist.append(height - window_radius)
        print(f"Número de ventanas: {len(rowlist)} filas x {len(collist)} columnas")

        window_count = 0
        for col in collist:
            windows = []
            rows_for_col = []
            for row in rowlist:
                window = img[row - window_radius:row + window_radius, col - window_radius:col + window_radius].copy()
                if window.shape[0] == window_radius * 2 and window.shape[1] == window_radius * 2:
                    windows.append(window)
                    rows_for_col.append(row)
                    window_count += 1
            
            if windows:
                windows = np.stack(windows).astype(np.float32)
                print(f"Procesando {len(windows)} ventanas en columna {col}")
                
                # Verificar rango de datos antes de predicción
                if np.abs(windows.min() - (-1.0)) > 0.1 or np.abs(windows.max() - 1.0) > 0.1:
                    print(f"ADVERTENCIA: Rango de ventana inusual - Min: {windows.min():.3f}, Max: {windows.max():.3f}")
                
                pred = session.run([output_name], {input_name: windows})[0]
                for i, row in enumerate(rows_for_col):
                    pred_mask = np.argmax(pred[i], axis=-1).astype(np.uint8)
                    if internal_window_radius < window_radius:
                        mm = rint(window_radius - internal_window_radius)
                        pred_mask = pred_mask[mm:-mm, mm:-mm]
                    output_mask[row - internal_window_radius:row + internal_window_radius,
                                col - internal_window_radius:col + internal_window_radius] = pred_mask

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