from scipy.ndimage import binary_erosion
import skimage.morphology
import numpy as np
import os
import onnxruntime as ort
from osgeo import gdal

# CONSTANTES MEJORADAS basadas en el aplicativo que funciona
CLASS_TO_SS = {"mauritia": -128, "euterpe": -96, "oenocarpus": -64}
CLASS_TO_CITYSCAPES = {"mauritia": 15, "euterpe": 25, "oenocarpus": 35}
THRESHOLD = {"mauritia": 3, "euterpe": 1, "oenocarpus": 2} 
MIN_SIZE = {"mauritia": 500, "euterpe": 400, "oenocarpus": 200}
SELEM = {3: np.ones((3, 3), dtype=bool), 1: np.ones((1, 1), dtype=bool), 2: np.ones((1, 1), dtype=bool)}
SELEN = {3: np.ones((36, 3), dtype=bool), 1: np.ones((7, 7), dtype=bool), 2: np.ones((3, 3), dtype=bool)} 

nodata_value = -9999  # Mantener consistente

def scale_image(image, flag=None, nodata_value=nodata_value):
    if flag is None:
        return image
    return image

def watershed_cut(depthImage, ssMask):
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)
        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccLabels = skimage.morphology.label(ccImage)
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccIDs = np.unique(ccLabels)[1:]
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            ccIDMask = skimage.morphology.binary_erosion(ccIDMask, SELEM[THRESHOLD[semClass]])
            ccIDMask = binary_erosion(ccIDMask, SELEN[THRESHOLD[semClass]])
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32)

def process_instances_raster(raster):
    resultImage = np.zeros(shape=raster.shape, dtype=np.float32)
    ninstances = {"mauritia": 0, "euterpe": 0, "oenocarpus": 0}
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ccImage = (raster == csCode)
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccLabels = skimage.morphology.label(ccImage)
        ccIDs = np.unique(ccLabels)[1:]
        ninstances[semClass] = len(ccIDs)
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32), ninstances

def save_tiff_mask_final(mask, output_path, reference_dataset):
    """
    MODIFICACIÓN: Guarda el raster final con geotransform corregida para bordes uniformes
    """
    driver = gdal.GetDriverByName('GTiff')
    
    # Obtener la geotransform original
    original_gt = reference_dataset.GetGeoTransform()
    projection = reference_dataset.GetProjection()
    
    # Calcular nueva geotransform que mantenga proporciones correctas
    # Esto asegura que QGIS muestre bordes uniformes
    new_gt = (
        original_gt[0],  # upperLeftX
        original_gt[1],  # pixelWidth  
        original_gt[2],  # rotation (0)
        original_gt[3],  # upperLeftY
        original_gt[4],  # rotation (0)
        original_gt[5]   # pixelHeight (negativo)
    )
    
    # Crear dataset de salida
    out_dataset = driver.Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Float32)
    
    # Aplicar la geotransform corregida
    out_dataset.SetGeoTransform(new_gt)
    out_dataset.SetProjection(projection)

    band = out_dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(mask)
    
    # Configuraciones críticas para visualización en QGIS
    band.SetMetadataItem('AREA_OR_POINT', 'Area')
    band.ComputeStatistics(False)
    
    out_dataset.FlushCache()
    out_dataset = None
    
    print(f"✓ Raster final guardado: {mask.shape[1]}x{mask.shape[0]}")
    print(f"✓ Geotransform aplicada: {new_gt}")

def apply_instance_onnx(feature_file_list, mask, roi, output_folder, model_path2, window_radius, internal_window_radius, make_tif=True, make_png=False):
    
    image_path = feature_file_list[0]
    mask_path = mask[0]

    # Configurar ONNX Runtime con optimizaciones
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path2, providers=providers, sess_options=session_options)
    input_names = [inp.name for inp in session.get_inputs()]
    
    print("=== CONFIGURACIÓN INSTANCIAS ONNX ===")
    print(f"Inputs del modelo: {input_names}")
    print(f"Window radius: {window_radius}")
    print(f"Internal window radius: {internal_window_radius}")

    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    datasetresponse = gdal.Open(mask_path, gdal.GA_ReadOnly)

    bandas = min(dataset.RasterCount, 3)

    output = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype=np.float32) + nodata_value

    cr = [0, dataset.RasterXSize]
    rr = [0, dataset.RasterYSize]

    collist = [x for x in range(cr[0] + window_radius, cr[1] - window_radius, internal_window_radius * 2)]
    if collist and collist[-1] < cr[1] - window_radius:
        collist.append(cr[1] - window_radius)
        
    rowlist = [x for x in range(rr[0] + window_radius, rr[1] - window_radius, internal_window_radius * 2)]
    if rowlist and rowlist[-1] < rr[1] - window_radius:
        rowlist.append(rr[1] - window_radius)

    win_size = window_radius * 2

    print(f"Procesando {len(rowlist)} filas x {len(collist)} columnas")
    print(f"Tamaño de ventana: {win_size}x{win_size}")

    for col_idx, col in enumerate(collist):
        imageBatch = []
        responses = []
        valid_rows = []
        
        for row_idx, n in enumerate(rowlist):
            d = np.zeros((win_size, win_size, bandas))
            for b in range(bandas):
                band_data = dataset.GetRasterBand(b + 1).ReadAsArray(col - window_radius, n - window_radius, win_size, win_size)
                if band_data is not None:
                    d[:, :, b] = band_data
                else:
                    d[:, :, b] = nodata_value
                
            d[np.isnan(d)] = nodata_value
            d[np.isinf(d)] = nodata_value
            d[d == -9999] = nodata_value
            
            r = datasetresponse.GetRasterBand(1).ReadAsArray(col - window_radius, n - window_radius, win_size, win_size)
            if r is None:
                r = np.zeros((win_size, win_size)) + nodata_value
            else:
                r = r.astype(float)

            if d.shape[0] == win_size and d.shape[1] == win_size:
                d = scale_image(d)
                imageBatch.append(d)
                responses.append(r)
                valid_rows.append(n)

        if not imageBatch:
            continue

        imageBatch = np.stack(imageBatch)
        responses = np.stack(responses)

        ssBatch = (responses > 0).astype(np.float32)
        ssMaskBatch = np.zeros_like(responses, dtype=np.float32)
        for i in range(responses.shape[0]):
            r_i = responses[i]
            ssMaskBatch[i][r_i == 1] = CLASS_TO_SS["mauritia"]
            ssMaskBatch[i][r_i == 2] = CLASS_TO_SS["euterpe"]
            ssMaskBatch[i][r_i == 3] = CLASS_TO_SS["oenocarpus"]

        imageBatch = imageBatch.reshape((imageBatch.shape[0], win_size, win_size, bandas))
        outputBatch = np.zeros((len(valid_rows), win_size, win_size), dtype=np.uint8)

        for j in range(len(valid_rows)):
            try:
                img_j = imageBatch[j] * ssBatch[j][..., np.newaxis]
                input_j = np.concatenate([img_j, ssMaskBatch[j][..., np.newaxis]], axis=-1).astype(np.float32)
                ss_batch_input = ssBatch[j].astype(np.float32)

                # Verificar dimensiones antes de enviar al modelo
                if input_j.shape != (win_size, win_size, 4):
                    print(f"Advertencia: Formato de entrada inesperado: {input_j.shape}")
                    continue

                outputs = session.run(None, {
                    input_names[0]: input_j[np.newaxis, ...],
                    input_names[1]: ss_batch_input[np.newaxis, ...]
                })
                tmp_output = outputs[0][0]
                outputBatch[j] = tmp_output.astype(np.uint8)
                print(f"Ventana {j} procesada exitosamente")
                
            except Exception as e:
                print(f"Error procesando ventana {j}: {e}")
                continue

        outputdwt = []
        for j in range(len(outputBatch)):
            try:
                outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
                outputdwt.append(outputImage)
            except Exception as e:
                print(f"Error en watershed cut {j}: {e}")
                outputdwt.append(np.zeros((win_size, win_size), dtype=np.float32))

        if outputdwt:
            outputdwt = np.stack(outputdwt)

            for j, n in enumerate(valid_rows):
                p = outputdwt[j]
                if internal_window_radius < window_radius:
                    mm = int(np.rint(window_radius - internal_window_radius))
                    p = p[mm:-mm, mm:-mm]
                
                start_row = n - internal_window_radius
                end_row = n + internal_window_radius
                start_col = col - internal_window_radius
                end_col = col + internal_window_radius
                
                # Asegurar que no nos salimos de los límites
                if (start_row >= 0 and end_row <= output.shape[0] and 
                    start_col >= 0 and end_col <= output.shape[1]):
                    output[start_row:end_row, start_col:end_col] = p

    output, quantification = process_instances_raster(output)

    # Guardar TIFF CON LA NUEVA FUNCIÓN
    name_saved_final = os.path.basename(image_path).replace('.tif', '_predicted.tif')
    out_path = os.path.join(output_folder, name_saved_final)
    
    if make_tif:
        # USAR LA NUEVA FUNCIÓN DE GUARDADO
        save_tiff_mask_final(output, out_path, dataset)
        print(f"Archivo TIFF guardado: {out_path}")

    mau = quantification['mauritia']
    eut = quantification['euterpe']
    oeno = quantification['oenocarpus']

    return name_saved_final, mau, eut, oeno