##### Script to detect 3 palm species using Deeplabv3 with ONNX ####

### Import libraries needed
import numpy as np
import os
from osgeo import gdal
import inspect
import warnings
from scipy.ndimage import binary_erosion
from skimage import morphology

from . import apply_model
from . import apply_model_dwt

# Suppress warnings
warnings.filterwarnings('ignore')

# GDAL exceptions
gdal.UseExceptions()

# Configure paths
cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
pluginPath = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, "trained_models"
    )
)

# NUEVO: Función de diagnóstico de imagen
def diagnostic_image_analysis(img_path):
    """Análisis detallado de la imagen para diagnóstico"""
    dataset = gdal.Open(img_path)
    if not dataset:
        return None, None
    
    info = {
        'size': (dataset.RasterXSize, dataset.RasterYSize),
        'bands': dataset.RasterCount,
        'data_type': gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType),
        'projection': dataset.GetProjection()[:50] + "..." if dataset.GetProjection() else "None"
    }
    
    # Analizar estadísticas por banda
    band_stats = []
    for i in range(1, min(4, dataset.RasterCount + 1)):
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        
        # Crear máscara de datos válidos
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = np.ones_like(data, dtype=bool)
        
        valid_data = data[valid_mask]
        
        stats = {
            'band': i,
            'min': np.min(valid_data) if len(valid_data) > 0 else 0,
            'max': np.max(valid_data) if len(valid_data) > 0 else 0,
            'mean': np.mean(valid_data) if len(valid_data) > 0 else 0,
            'std': np.std(valid_data) if len(valid_data) > 0 else 0,
            'nodata_value': nodata,
            'nodata_pixels': np.sum(~valid_mask),
            'data_range': None
        }
        
        # Calcular rango de datos
        if len(valid_data) > 0:
            stats['data_range'] = stats['max'] - stats['min']
            
        band_stats.append(stats)
    
    dataset = None
    
    print("=== DIAGNÓSTICO DE IMAGEN ===")
    print(f"Archivo: {os.path.basename(img_path)}")
    print(f"Tamaño: {info['size']}")
    print(f"Bandas: {info['bands']}")
    print(f"Tipo de datos: {info['data_type']}")
    print("Estadísticas por banda:")
    for stats in band_stats:
        print(f"  Banda {stats['band']}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
              f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, nodata={stats['nodata_pixels']}")
    
    return info, band_stats

# NUEVO: Preprocesamiento mejorado
def improved_preprocessing(image_data, original_dtype, nodata_value, scaling='normalize'):
    """
    Preprocesamiento mejorado para diferentes tipos de datos y rangos dinámicos
    """
    # Crear máscara de nodata
    nodata_mask = np.all(image_data == nodata_value, axis=2) if nodata_value is not None else np.zeros((image_data.shape[0], image_data.shape[1]), dtype=bool)
    valid_mask = ~nodata_mask
    
    # Convertir a float32 para procesamiento
    image_float = image_data.astype(np.float32)
    
    print(f"Preprocesamiento: tipo_dato={original_dtype}, scaling={scaling}")
    
    if scaling == 'normalize':
        # PREPROCESAMIENTO MEJORADO BASADO EN EL RANGO DINÁMICO
        for b in range(image_float.shape[2]):
            band_data = image_float[:, :, b]
            valid_pixels = band_data[valid_mask]
            
            if len(valid_pixels) > 0:
                # Usar percentiles para evitar outliers
                p1, p99 = np.percentile(valid_pixels, [1, 99])
                
                # Si el rango es muy pequeño, usar min/max en su lugar
                if p99 - p1 < 10:  # Rango muy pequeño
                    p1, p99 = np.percentile(valid_pixels, [0.5, 99.5])
                
                print(f"  Banda {b+1}: percentiles 1-99% = [{p1:.2f}, {p99:.2f}]")
                
                # Aplicar estiramiento de contraste mejorado
                band_stretched = np.zeros_like(band_data)
                if p99 > p1:
                    # Escalar a [0, 1] usando percentiles
                    band_stretched[valid_mask] = np.clip((valid_pixels - p1) / (p99 - p1), 0, 1)
                    
                    # Mejorar contraste con ajuste gamma
                    band_stretched[valid_mask] = np.power(band_stretched[valid_mask], 0.8)
                    
                    # Escalar a [-1, 1] para el modelo
                    image_float[:, :, b] = band_stretched * 2 - 1
                else:
                    # Si no hay variación, normalizar simple
                    image_float[:, :, b] = (band_data - np.mean(valid_pixels)) / (np.std(valid_pixels) + 1e-8)
            else:
                # Si no hay píxeles válidos, mantener como está
                image_float[:, :, b] = band_data / 127.5 - 1
        
        # Restaurar valores nodata
        for b in range(image_float.shape[2]):
            image_float[nodata_mask, b] = -1
            
        print(f"  Rango final: min={image_float.min():.3f}, max={image_float.max():.3f}")
        
    elif scaling == 'mean_std':
        # Escalado mean-std tradicional
        nodata_mask_3d = np.repeat(nodata_mask[:, :, np.newaxis], image_float.shape[2], axis=2)
        valid_mask_3d = ~nodata_mask_3d
        
        for b in range(image_float.shape[2]):
            valid_pixels = image_float[:, :, b][valid_mask]
            if len(valid_pixels) > 0:
                mean_val = np.mean(valid_pixels)
                std_val = np.std(valid_pixels)
                if std_val > 0:
                    image_float[:, :, b][valid_mask] = (image_float[:, :, b][valid_mask] - mean_val) / std_val
    
    return image_float

# NUEVO: Postprocesamiento mejorado
def postprocess_mask(mask, min_region_size=30):
    """
    Postprocesamiento mejorado para eliminar ruido
    """
    try:
        cleaned_mask = mask.copy()
        
        # Procesar cada clase de palmas
        for class_id in [1, 2, 3]:
            class_mask = mask == class_id
            if np.sum(class_mask) > 0:  # Solo procesar si hay píxeles de esta clase
                # Eliminar regiones muy pequeñas
                class_mask_cleaned = morphology.remove_small_objects(class_mask, min_size=min_region_size)
                # Eliminar hoyos pequeños
                class_mask_cleaned = morphology.remove_small_holes(class_mask_cleaned, area_threshold=min_region_size)
                
                # Actualizar máscara
                cleaned_mask[class_mask & ~class_mask_cleaned] = 0
                cleaned_mask[class_mask_cleaned] = class_id
                    
        return cleaned_mask
    except ImportError:
        print("ADVERTENCIA: skimage no disponible, saltando postprocesamiento")
        return mask

### Main Plugin Function ###
def apply_palmeras(INPUT_RASTER, OUTPUT_RASTER,
                   window_radius=256, internal_window_radius=None, batch_size=0,
                   crown_dilation=0, crown_separation=0, blend_seams=True,
                   low_memory=False, memory_budget_mb=0,
                   fill_holes=True):
    """Detecta palmeras en INPUT_RASTER y deja el resultado en OUTPUT_RASTER.

    window_radius e internal_window_radius controlan el ventaneo de la
    segmentacion semantica. Con los valores por defecto (256 y None) el
    comportamiento es identico al de siempre: la ventana mide 512 px y se
    conserva el 75 % central. La etapa de instancias (DWT) mantiene sus propios
    radios, 350 y 262.

    batch_size limita cuantas ventanas viajan juntas a cada session.run. Con 0 se
    envia la columna entera, como siempre; un valor mas bajo recorta el pico de
    memoria de forma proporcional, a cambio de mas llamadas al modelo.

    crown_dilation dilata cada copa detectada esos METROS. Con 0 no se toca nada.
    No altera el recuento de individuos, solo su superficie.

    crown_separation separa copas que se tocan: es el radio en pixeles con el que se
    erosionan los marcadores del watershed. Con 0 se mantiene el comportamiento
    historico, que no logra partir copas pegadas por un frente ancho.

    blend_seams combina las ventanas solapadas de la segmentacion semantica en vez de
    pisar unas con otras. Quita las costuras rectas del mosaico a cambio de un
    acumulador de alto x ancho x clases en float32.

    low_memory proyecta lo que va a costar la segmentacion y, si no cabe en el
    presupuesto, la parte en franjas de filas que se liberan una tras otra.
    memory_budget_mb fija ese presupuesto; con 0 se toma el 60 % de la memoria libre.
    """
    ### Model settings
    window_radius = int(window_radius)
    output_folder = os.path.dirname(OUTPUT_RASTER) if OUTPUT_RASTER != 'TEMPORARY_OUTPUT' else os.path.join(os.path.dirname(INPUT_RASTER), 'output')
    feature_file_list = [INPUT_RASTER]
    if internal_window_radius:
        internal_window_radius = max(1, min(window_radius, int(internal_window_radius)))
    else:
        internal_window_radius = int(round(window_radius * 0.75))
    model_path = os.path.join(pluginPath, "model_deeplabv3_segmentation_v1.onnx")
    model_path2 = os.path.join(pluginPath, "model_dwt_instance_segmenetation_v1.onnx")
    
    print("=== CONFIGURACIÓN DE MODELOS ===")
    print(f"Modelo segmentación: {model_path}")
    print(f"Modelo instancias: {model_path2}")
    print(f"Window radius: {window_radius}")
    print(f"Internal window radius: {internal_window_radius}")

    ### Verificar si los modelos existen
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo ONNX no se encuentra en: {model_path}")
    if not os.path.exists(model_path2):
        raise FileNotFoundError(f"El modelo ONNX no se encuentra en: {model_path2}")

    ### Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # NUEVO: Ejecutar diagnóstico de imagen
    print("=== EJECUTANDO DIAGNÓSTICO DE IMAGEN ===")
    info, band_stats = diagnostic_image_analysis(INPUT_RASTER)
    
    ### Semantic segmentation con configuración mejorada
    name_saved = apply_model.apply_semantic_segmentation_onnx(
        input_file_list=feature_file_list,
        output_folder=output_folder,
        model_path=model_path,
        window_radius=window_radius,
        internal_window_radius=internal_window_radius,
        make_tif=True,
        scaling='normalize',  # Usar normalización mejorada
        batch_size=batch_size,
        blend_seams=blend_seams,
        low_memory=low_memory,
        memory_budget_mb=memory_budget_mb
    )

    ### Procesamiento de instancias
    window_radius_instances = 350  
    internal_window_radius_instances = int(round(window_radius_instances * 0.75))
    name_mask_clas = os.path.join(output_folder, name_saved)
    mask = [name_mask_clas]
    roi = []

    print("=== PROCESANDO INSTANCIAS ===")
    print(f"Window radius instancias: {window_radius_instances}")
    print(f"Internal window radius instancias: {internal_window_radius_instances}")
    print(f"Crown dilation: {crown_dilation} m")
    print(f"Crown separation: {crown_separation} px")

    name_saved_final, mau, eut, oeno = apply_model_dwt.apply_instance_onnx(
        feature_file_list,
        mask,
        roi,
        output_folder,
        model_path2,
        window_radius_instances,
        internal_window_radius_instances,
        make_tif=True,
        make_png=False,
        crown_dilation=crown_dilation,
        crown_separation=crown_separation,
        fill_holes=fill_holes
    )
    
    # Salidas con nombre propio junto al destino elegido en Processing. El raster de
    # instancias se copia ademas a OUTPUT_RASTER porque ese es el destino que QGIS
    # reserva para el parametro; si ya coinciden, no se duplica.
    import shutil

    dest_dir = os.path.dirname(OUTPUT_RASTER) or output_folder
    individuos = os.path.join(dest_dir, 'detected_individuals.tif')
    cobertura = os.path.join(dest_dir, 'detected_coverage.tif')

    out_imag = os.path.join(output_folder, name_saved_final)
    if os.path.exists(individuos):
        os.remove(individuos)
    os.rename(out_imag, individuos)

    if os.path.exists(cobertura):
        os.remove(cobertura)
    os.rename(name_mask_clas, cobertura)

    if os.path.abspath(individuos) != os.path.abspath(OUTPUT_RASTER):
        shutil.copyfile(individuos, OUTPUT_RASTER)

    print("=== FINAL RESULTS ===")
    print(f"Detected individuals: {individuos}")
    print("   Float32, nodata 0 (transparent). Pixel values: "
          "15 = Mauritia flexuosa, 25 = Euterpe precatoria, 35 = Oenocarpus bataua")
    print(f"Detected coverage: {cobertura}")
    print("   Byte, nodata 0. Pixel values: 0 = background, 1 = Mauritia flexuosa, "
          "2 = Euterpe precatoria, 3 = Oenocarpus bataua")
    print(f"Processing destination: {OUTPUT_RASTER}")
    print(f"Mauritia flexuosa: {mau} palms")
    print(f"Euterpe precatoria: {eut} palms")
    print(f"Oenocarpus bataua: {oeno} palms")

    return OUTPUT_RASTER, cobertura, mau, eut, oeno
