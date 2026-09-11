from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           center_of_mass, distance_transform_edt, find_objects)
from scipy.ndimage import maximum as ndi_maximum
import skimage.morphology
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed
import numpy as np
import os
import onnxruntime as ort
from osgeo import gdal

# CONSTANTES MEJORADAS basadas en el aplicativo que funciona
CLASS_TO_SS = {"mauritia": -128, "euterpe": -96, "oenocarpus": -64}
CLASS_TO_CITYSCAPES = {"mauritia": 15, "euterpe": 25, "oenocarpus": 35}
# Codigos de la mascara semantica (*_argmax.tif), que es la que manda sobre donde
# hay palmera de cada especie y donde no.
CLASS_TO_SEMANTIC = {"mauritia": 1, "euterpe": 2, "oenocarpus": 3}
THRESHOLD = {"mauritia": 3, "euterpe": 1, "oenocarpus": 2} 
# Area minima para que una pieza cuente como copa, en PIXELES. Solo se usa cuando el
# raster no esta georreferenciado; con tamano de pixel se prefiere MIN_CROWN_DIAMETER_M,
# que no depende de la resolucion.
MIN_SIZE = {"mauritia": 500, "euterpe": 400, "oenocarpus": 200}
# Diametro minimo de copa, en metros. A 4,47 cm/px los 500 px de Mauritia son 1,0 m2,
# es decir una copa de 1,1 m: cualquier fleco del borde de la mascara pasaba el filtro.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha de aguajal:
#
#   diametro min   copas/ha  aciertos  partidas  fundidas  sobrantes  F1   cobertura
#      1,1 m (antes)  102       39         3         7        52     0,50   94,8 %
#      2,0 m           81       39         3         7        32     0,57   93,0 %
#      2,5 m           75       40         2         7        27     0,62   91,8 %
#      3,0 m           72       39         2         7        25     0,61   90,7 %
#      4,0 m           69       38         2         7        23     0,60   89,4 %
#
# A 2,5 m se van la mitad de las instancias sobrantes sin perder ni una copa
# etiquetada, y el conteo pasa de 102 a 75 copas/ha frente a las 57 reales. Por encima
# de 3 m empiezan a caer aciertos. Las otras dos especies se escalan por el mismo
# factor sobre su separacion tipica (2,5 / 3,0 = 0,83).
MIN_CROWN_DIAMETER_M = {"mauritia": 2.5, "euterpe": 1.4, "oenocarpus": 1.9}
# Erosion que separa copas pegadas: una sola, explicita, por especie.
#
# Antes eran dos erosiones encadenadas -una de skimage con SELEM y otra de scipy con
# SELEN-, ambas indexadas por THRESHOLD en vez de por especie. Encadenar dos erosiones
# equivale a erosionar una sola vez por la suma de Minkowski de los dos elementos, asi
# que aqui va ya compuesta:
#
#   Mauritia    (3,3) + (3,3) -> (5,5)
#   Euterpe     (1,1) + (7,7) -> (7,7)   el (1,1) no hacia nada
#   Oenocarpus  (1,1) + (3,3) -> (3,3)   idem
#
# El (1,1) en dos de las tres especies era una erosion nula: solo Mauritia recibia las
# dos de verdad. Y su segundo elemento era np.ones((36, 3)), que recortaba unos 18 px
# por arriba y por abajo y solo 1 px por los lados: a 4 cm/px, ~70 cm en vertical frente
# a 8 cm en horizontal, sobre una copa que es aproximadamente radial.
#
# Indexar por especie y no por THRESHOLD evita ademas que dos especies con el mismo
# umbral compartan elemento sin quererlo.
EROSION = {
    "mauritia": np.ones((5, 5), dtype=bool),
    "euterpe": np.ones((7, 7), dtype=bool),
    "oenocarpus": np.ones((3, 3), dtype=bool),
}

nodata_value = -9999  # Mantener consistente

# Salidas de depuracion. Con False no se escribe nada y no cuestan memoria: los tres
# rasteres se calculan solo si se piden. Ponlo a True para ver por dentro la separacion
# de copas; se escriben junto al raster de instancias, con estos sufijos:
#
#   _dwt_crudo.tif      el mapa de profundidad tal cual sale del modelo (0..15)
#   _dwt_crudo_ssmask   la mascara semantica que se le paso
#   _dwt_filtrado.tif   solo el umbral, antes de cortar
#   _dwt_copas.tif      las copas ya cortadas: lo que se escribe como instancias
#
# El mosaico de profundidad y el de mascara se construyen siempre, porque la
# separacion los necesita; lo que ahorra el False es el calculo y la escritura de los
# tres ficheros derivados.
DEBUG_RASTERS = False

def scale_image(image, flag=None, nodata_value=nodata_value):
    if flag is None:
        return image
    return image

# Escala de las copas -> parametros de la separacion automatica.
#
# Va en METROS y se pasa a pixeles con el tamano de pixel del raster. Antes se
# deducia de la propia mascara con una transformada de distancia, y no era estable:
# donde varias copas salian pegadas, el circulo inscrito era el de la mancha entera
# y no el de una copa. Sobre el mismo aguajal, una ventana daba 65 px de separacion
# y otra 32, con lo que la calibracion no se trasladaba de una a otra.
#
# El valor de Mauritia sale de la calibracion visual sobre una ventana real: 65 px a
# 4,47 cm/px son 2,9 m. Los otros dos se escalan por tamano de copa.
MIN_CROWN_SPACING_M = {"mauritia": 3.0, "euterpe": 1.7, "oenocarpus": 2.3}
# El mapa de profundidad del DWT llega en 0..15 y tiene una MESETA en su parte alta:
# dentro de cada copa hay una zona plana de valor 12-13 que ocupa buena parte de ella,
# y entre dos copas pegadas el valor baja. Quedarse con esa meseta da un nucleo por
# copa; el problema no es encontrarlos sino que dos nucleos vecinos siguen unidos por
# un puente fino, que se corta con una erosion pequena.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha de aguajal, contando
# tambien las instancias sobrantes -sin esa columna gana siempre el metodo que mas
# instancias genera, porque cuantas mas produce mas copas toca por casualidad-:
#
#   marcadores                      copas/ha  aciertos  partidas  fundidas  IoU   F1
#   brillo, prominencia 0,08            77       23        10        16     0,46  0,35
#   simetria radial, prominencia 0,25   64       18        11        17     0,43  0,30
#   meseta >= 12, erosion 2 px          86       37         4         9     0,61  0,52
#   meseta + regla de mancha            84       39         2         9     0,63  0,56
#
# La verdad de campo son 57 copas/ha. Los metodos de relieve -suavizar una superficie
# y buscar sus maximos- se quedan en la mitad de aciertos porque cada hoja iluminada
# es un maximo del brillo y el mapa de profundidad da una meseta por grupo, no por
# copa; la meseta pregunta directamente lo que el modelo aprendio.
# Umbral del filtro: que parte del mapa de profundidad se queda como copa. El mismo
# para todas las manchas, sea cual sea su forma.
#
# Desde que las semillas no se hacen crecer, este umbral decide directamente el TAMANO
# de la copa que se escribe, no solo por donde se corta. Por eso conviene mas bajo de
# lo que convenia antes: con 9 las copas salian al 68 % de la cobertura, con 6 al 75 %,
# y el IoU sube en consecuencia.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha:
#
#   umbral  copas/ha  aciertos  partidas  fundidas  IoU   F1   cobertura
#     5        82        45         1         4    0,60  0,66    79,4 %
#     6        85        47         1         2    0,57  0,67    75,1 %
#     7        86        46         0         4    0,55  0,65    72,4 %
#     8        81        44         0         6    0,52  0,65    69,6 %
#     9        82        48         0         2    0,53  0,70    67,7 %
#
# A 6 las copas quedan mas cerca de su tamano real y las fusiones se mantienen en 2, a
# cambio de una copa partida. Por debajo de 6 el nucleo engorda tanto que los cuellos
# vuelven a ser anchos y las fusiones suben.
CORE_DEPTH_LARGE = 6            # umbral en manchas grandes y compactas
CORE_DEPTH_SMALL = 6            # umbral en manchas pequenas, alargadas o delgadas
# Radio en pixeles del disco con el que se ABRE el nucleo de una mancha grande, para
# cortar los cuellos estrechos que unen dos copas vecinas.
#
# Apertura, no erosion. La erosion quita una banda de r pixeles de todo el contorno:
# se come el cuerpo de la copa igual que el cuello, y con r grande no queda nucleo. La
# apertura es esa erosion seguida de una dilatacion por el mismo disco, de modo que el
# resultado es la union de todos los discos de radio r que caben dentro de la mancha:
# un cuello mas estrecho que 2r no admite ningun disco y desaparece, mientras que el
# cuerpo, que si los admite, vuelve a su tamano. Es cortar solo los conectores.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha de aguajal (4,47 cm/px),
# con la ultima columna = cuanto del nucleo sobrevive a la operacion:
#
#   operacion        radio  copas/ha  aciertos  partidas  fundidas  IoU   F1   nucleo
#   erosion            10      109       36         6         7    0,59  0,44  31,9 %
#   erosion            20       96       35         3        11    0,58  0,46  11,1 %
#   erosion            50       92       33         3        13    0,57  0,45   5,5 %
#   apertura            6      108       36         2        11    0,59  0,44  73,4 %
#   apertura            8      101       39         3         7    0,61  0,50  67,6 %
#   apertura           10       99       39         3         7    0,62  0,51  64,0 %
#   apertura           12       98       39         3         7    0,62  0,51  61,4 %
#   apertura           18       96       35         4        10    0,59  0,46  51,8 %
#
# La verdad de campo son 57 copas/ha. La apertura gana a la erosion en todos los radios
# y conserva el doble de nucleo. Entre 8 y 12 el resultado es el mismo, asi que 10 deja
# margen por los dos lados; a 4,47 cm/px son 45 cm, el ancho de cuello que se corta.
# Como se separan dos copas que comparten nucleo.
#
# Erosionar o abrir el nucleo QUITA material, y ese es el problema: con radio pequeno
# no llega a cortar cuellos anchos, y con radio grande se come el cuerpo de la copa o
# la parte por un estrechamiento que estaba dentro de una misma palmera.
#
# Lo que hace falta no es quitar, es CORTAR: una linea por el punto mas estrecho de la
# union, dejando los dos cuerpos enteros. Eso es un watershed sobre la transformada de
# distancia del propio nucleo: la distancia al borde es maxima en el centro de cada
# copa y minima en el cuello, asi que sembrar en los maximos y dejar crecer reparte el
# nucleo entre ellos y la frontera cae justo en el estrangulamiento. No se pierde ni un
# pixel salvo la linea de separacion.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha, todo con los huecos internos
# ya rellenados:
#
#   metodo                       copas/ha  aciertos  partidas  fundidas  IoU   F1
#   sin cortar                       60       34         1        15    0,55  0,59
#   apertura de 8 px                 66       37         1        12    0,60  0,61
#   corte por cuello                 76       45         1         4    0,66  0,69
#   apertura + corte por cuello      76       45         1         4    0,66  0,69
#
# El corte por cuello hace todo el trabajo: anadirle la apertura encima no cambia un
# solo numero, asi que la apertura sale del codigo.
# Como se encuentran las semillas dentro del nucleo, para saber por donde cortarlo.
#
# Se usa la PROMINENCIA de los maximos de la transformada de distancia, no la distancia
# minima entre picos. La diferencia importa: pidiendo que dos picos esten a N pixeles,
# dos copas cuyos centros estan mas juntos que eso dan un solo pico y no hay forma de
# separarlas. La prominencia pregunta otra cosa -cuanto sobresale cada lobulo sobre el
# valle que lo une al vecino- y da una semilla por lobulo real por juntos que esten.
# De que el corte sea razonable ya se encarga justified_cut.
#
# El valor son pixeles de radio: un lobulo cuenta si su maximo de distancia supera en 8
# px al cuello que lo une al de al lado. A 4,47 cm/px son 36 cm.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha:
#
#   busqueda de semillas        copas/ha  aciertos  partidas  fundidas  IoU   F1
#   distancia minima 40 px          80       48         0         2    0,52  0,71
#   distancia minima 30 px          83       47         0         2    0,51  0,68
#   distancia minima 20 px          86       43         0         2    0,47  0,61
#   prominencia 3 px                83       47         0         2    0,49  0,68
#   prominencia 5 px                80       47         0         2    0,52  0,70
#   prominencia 8 px                79       48         0         2    0,53  0,72
#
# Apretar la distancia minima solo empeora: hace trizas el nucleo y la guarda acaba
# rechazandolo todo. Las 2 fusiones que quedan no son cortables por ningun ajuste: son
# copas tan solapadas que su nucleo es un bulto redondo con un unico maximo.
# Bajado de 8 a 5 px para cortar algo mas. Sobre las 56 copas etiquetadas da lo mismo
# -47 aciertos, 1 partida, 2 fusiones con 8, 6, 5 y 4- porque los cuellos que quedan no
# son cortables a ningun radio, pero acepta lobulos menos marcados y en otras imagenes
# encuentra cortes que a 8 se pasaba. Lo que impide que se pase de la raya no es este
# numero sino justified_cut.
NECK_PROMINENCE_PX = 5.0
NECK_MIN_PIECE = 0.30           # un trozo cuenta si llega a esta fraccion de copa
# Una pasada mas de corte sobre las piezas que siguen teniendo area para mas de
# NECK_SECOND_ABOVE copas, buscando lobulos menos prominentes que en la primera.
NECK_SECOND_PROMINENCE = 4.0
NECK_SECOND_ABOVE = 1.0
# Area maxima de un hueco interior que se tapa dentro de una copa, en copas tipicas.
# A 4,47 cm/px, 0,05 copas son 1,4 m2: un claro entre hojas, no un espacio entre
# palmeras.
#
# Solo se tapa el hueco rodeado por UNA sola copa. El que toca dos es el espacio que
# las separa, y taparlo las volveria a unir: probado rellenando pieza a pieza sin esa
# comprobacion, los aciertos caian de 47 a 44 y las fusiones subian de 2 a 4.
#
# Con la comprobacion puesta se tapan 1167 huecos, la separacion no se mueve -47
# aciertos, 1 partida, 2 fusiones- y la cobertura sube del 75,1 % al 76,0 %.
CROWN_HOLE_MAX = 0.05
# Erosion del nucleo ANTES de cortar, en pixeles. Sin dilatacion de vuelta: encoge la
# copa a proposito. Adelgaza los cuellos entre copas vecinas, y asi el corte los
# encuentra donde antes no llegaba.
#
# Es la operacion que se habia descartado, pero el papel ha cambiado. Cuando las
# semillas crecian hasta llenar la mascara, erosionar solo desplazaba el marcador y
# estropeaba la forma. Ahora la semilla ES la copa, y encoger un poco a cambio de
# separar mejor es el intercambio que interesa.
#
# Medido contra 56 copas etiquetadas a mano sobre 0,99 ha, erosionando antes de cortar:
#
#   erosion  copas/ha  aciertos  partidas  fundidas  sobrantes  IoU   F1   cobertura
#     0 px      85        47         1         2        34     0,57  0,67   75,2 %
#     1 px      84        48         0         2        34     0,55  0,69   71,9 %
#     2 px      81        48         0         2        31     0,54  0,71   68,7 %
#     3 px      76        43         0         6        30     0,49  0,66   65,0 %
#     5 px      75        41         1         2        30     0,39  0,63   51,7 %
#
# A 2 px desaparece la copa partida que quedaba, se van 3 instancias sobrantes y el F1
# sube a 0,71, el mejor medido. Cuesta 6 puntos de cobertura. A 3 px el nucleo empieza
# a romperse y las fusiones se triplican. Erosionar DESPUES de cortar es peor en todo:
# 45 aciertos y vuelve la copa partida.
#
# Con 1 px se conservan 3 puntos mas de cobertura con la misma separacion, si el area
# por individuo importa mas que las instancias sobrantes.
CORE_EROSION_PX = 2
BLOB_SINGLE_CROWN = 3.0
BLOB_ELONGATED = 2.0            # largo / ancho
MIN_PIXEL_SIZE_M = 0.001        # por debajo de esto el raster no esta en metros
CROWN_RADIUS_PERCENTILE = 70    # solo para el respaldo sin georreferencia
FALLBACK_SPACING_FACTOR = 1.6   # separacion en radios de copa, si no hay metros


def min_crown_size(semClass, pixel_size):
    """Area minima de una copa, en pixeles.

    Con tamano de pixel en metros se calcula desde MIN_CROWN_DIAMETER_M, para que el
    filtro signifique lo mismo a cualquier resolucion. Sin el -un JPG suelto- se cae al
    valor en pixeles de siempre.
    """
    if pixel_size and pixel_size >= MIN_PIXEL_SIZE_M:
        diameter = MIN_CROWN_DIAMETER_M.get(semClass, 2.0)
        return max(1, int(round(np.pi * (diameter / 2.0) ** 2 / (pixel_size ** 2))))
    return MIN_SIZE.get(semClass, 200)


def crown_radius(region):
    """Radio tipico de las copas de una clase, en pixeles.

    Respaldo para rasteres sin tamano de pixel en metros. Mide la distancia al
    borde de la mascara y se queda con la cola alta, que en una mancha de copas
    sueltas es el radio de copa. Cuando varias copas salen pegadas lo sobreestima,
    y por eso solo se usa a falta de algo mejor.
    """
    distance = distance_transform_edt(region)
    inside = distance[region]
    if inside.size == 0:
        return 0.0
    core = inside[inside > np.percentile(inside, CROWN_RADIUS_PERCENTILE)]
    if core.size == 0:
        return float(inside.max())
    return float(np.median(core))


def crown_spacing(semClass, region, pixel_size):
    """Separacion minima entre copas de esa especie, en pixeles.

    El suelo sale de MIN_SIZE, el area minima que el propio proceso acepta como copa:
    una copa de esa area tiene radio sqrt(MIN_SIZE / pi), y pedir marcadores mas
    juntos que eso no tiene sentido, porque lo que salga entre ellos se descarta
    despues por pequeno. Importa sobre todo cuando el raster no esta georreferenciado
    -un JPG suelto, por ejemplo- y hay que estimar la escala de la propia mascara:
    en una ventana casi vacia esa estimacion se va a cero y, sin suelo, la separacion
    caia a 3 px y hacia trizas las copas.
    """
    floor = max(3, int(round((min_crown_size(semClass, pixel_size) / np.pi)
                            ** 0.5)))
    if pixel_size and pixel_size >= MIN_PIXEL_SIZE_M:
        metres = MIN_CROWN_SPACING_M.get(semClass, 2.0)
        return max(floor, int(round(metres / pixel_size)))
    return max(floor, int(round(crown_radius(region) * FALLBACK_SPACING_FACTOR)))


def justified_cut(labels, typical, min_distance):
    """Los trozos del corte son copas distintas, o es una copa rota?

    Dos palmeras no pueden tener los centros mas juntos que la separacion minima, y un
    trozo que no llega a NECK_MIN_PIECE de una copa no es una copa: es una esquirla.
    """
    sizes = np.bincount(labels.ravel())[1:]
    good = np.nonzero(sizes >= NECK_MIN_PIECE * typical)[0]
    if good.size < 2:
        return False
    centres = [np.array(center_of_mass(labels == k + 1)) for k in good]
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if float(np.hypot(*(centres[i] - centres[j]))) < min_distance:
                return False
    return True


def cut_oversized(cores, spacing, typical):
    """Segunda pasada de corte sobre las piezas que aun dan para varias copas."""
    labels = skimage.morphology.label(cores)
    if labels.max() == 0:
        return cores
    result = cores.copy()
    for cc, box in enumerate(find_objects(labels), start=1):
        if box is None:
            continue
        piece = (labels[box] == cc)
        if piece.sum() < NECK_SECOND_ABOVE * typical:
            continue
        cut = cut_necks(piece, spacing, typical, NECK_SECOND_PROMINENCE)
        if cut is not piece:
            window = result[box]
            window[piece & ~cut] = False
    return result


def fill_own_holes(mask, limit):
    """Tapa los huecos interiores que pertenecen a una sola copa.

    Un hueco rodeado por dos copas distintas es el espacio entre ellas; taparlo las
    soldaria. Solo se tapa el que tiene una unica copa alrededor.
    """
    holes = binary_fill_holes(mask) & ~mask
    if not holes.any():
        return mask
    labels = skimage.morphology.label(mask)
    marked = skimage.morphology.label(holes)
    result = mask.copy()
    disc = skimage.morphology.disk(1)
    for cc, box in enumerate(find_objects(marked), start=1):
        if box is None:
            continue
        hole = (marked[box] == cc)
        if hole.sum() > limit:
            continue
        ring = skimage.morphology.dilation(hole, disc) & ~hole
        owners = set(np.unique(labels[box][ring]).tolist()) - {0}
        if len(owners) == 1:
            result[box][hole] = True
    return result


def cut_necks(core, spacing, typical, prominence=None):
    """Parte el nucleo por sus estrangulamientos, sin quitarle material.

    Si el corte no queda justificado se devuelve el nucleo intacto: es preferible una
    copa fusionada a una copa partida por la mitad.
    """
    if prominence is None:
        prominence = NECK_PROMINENCE_PX
    distance = distance_transform_edt(core).astype(np.float32)
    # h_maxima deja solo los maximos que sobresalen NECK_PROMINENCE_PX sobre su
    # entorno, y fusiona en uno los que no. Asi cada lobulo aporta una semilla, por
    # cerca que este del vecino.
    tops = skimage.morphology.h_maxima(distance, prominence) > 0
    seeds = skimage.morphology.label(tops & core)
    if seeds.max() < 2:
        del distance, tops
        return core
    pieces = watershed(-distance, seeds, mask=core)
    pieces[find_boundaries(pieces, mode='inner', background=0)] = 0
    del distance, seeds
    labels = skimage.morphology.label(pieces > 0)
    if labels.max() < 2 or not justified_cut(labels, typical, spacing):
        return core
    return labels > 0


def crown_filter(depthImage, ssMask, pixel_size):
    """El mapa de profundidad solo filtrado: el umbral, sin cortar nada.

    Por cada mancha de la mascara semantica se aplica el umbral que le toca segun su
    forma -CORE_DEPTH_LARGE si es compacta y da para varias copas, CORE_DEPTH_SMALL en
    los demas casos- y se devuelve eso tal cual. Sin apertura y sin cortado.

    Es el paso anterior al corte: comparandolo con el raster de copas se ve que le
    quita el cortado al nucleo. Los pixeles llevan el codigo de especie.
    """
    result = np.zeros(shape=ssMask.shape, dtype=np.float32)
    for semClass, csCode in CLASS_TO_CITYSCAPES.items():
        region = (ssMask == CLASS_TO_SS[semClass])
        if not region.any():
            continue
        spacing = crown_spacing(semClass, region, pixel_size)
        typical = np.pi * float(spacing) ** 2
        for prop in regionprops(skimage.morphology.label(region)):
            y0, x0, y1, x1 = prop.bbox
            local = prop.image
            minor = max(1.0, prop.axis_minor_length)
            compact = prop.axis_major_length / minor < BLOB_ELONGATED
            large = prop.area >= BLOB_SINGLE_CROWN * typical and compact
            level = CORE_DEPTH_LARGE if large else CORE_DEPTH_SMALL
            window = result[y0:y1, x0:x1]
            window[(depthImage[y0:y1, x0:x1] >= level) & local] = csCode
    return result


def crown_markers(depthImage, region, spacing, pixel_size):
    """Un marcador por copa, a partir de la meseta del mapa de profundidad.

    Tres casos segun la forma de la mancha:

      compacta y sin area para dos copas  -> es UNA copa, no se subdivide
      compacta y con area para varias     -> umbral CORE_DEPTH_LARGE
      alargada                            -> umbral CORE_DEPTH_SMALL

    En los dos ultimos, al nucleo se le tapan los huecos internos y se le cortan los
    cuellos con cut_necks, que separa por el estrangulamiento sin quitar material. Si
    no queda nucleo, la mancha entera cuenta como una copa.

    pixel_size se conserva en la firma por compatibilidad con quien la llama.
    """
    typical = np.pi * float(spacing) ** 2      # area de una copa tipica, en pixeles
    blobs = skimage.morphology.label(region)
    markers = np.zeros(region.shape, dtype=np.int32)
    nxt = 1
    for prop in regionprops(blobs):
        y0, x0, y1, x1 = prop.bbox
        local = prop.image
        minor = max(1.0, prop.axis_minor_length)
        elongation = prop.axis_major_length / minor
        compact = elongation < BLOB_ELONGATED
        large = prop.area >= BLOB_SINGLE_CROWN * typical and compact
        cores = (depthImage[y0:y1, x0:x1]
                 >= (CORE_DEPTH_LARGE if large else CORE_DEPTH_SMALL)) & local
        if compact and not large:
            # Compacta y sin area para dos copas: es UNA copa, aunque mida 8 o 10 m.
            #
            # Pero manda el filtrado. Si el umbral ya ha dejado dos nucleos separados y
            # creibles -tamano de copa y centros a mas de la separacion minima- marcar
            # la mancha entera los volveria a soldar, y salian unidas en la salida dos
            # copas que en el filtrado estaban sueltas.
            pieces = skimage.morphology.label(cores)
            if pieces.max() < 2 or not justified_cut(pieces, typical, spacing):
                # Se marca la mancha entera, pero pasandola igualmente por el corte:
                # un brazo fino que une dos copas dentro de la misma mancha tiene que
                # cortarse tambien aqui. Sin esto, estas manchas se saltaban el corte
                # por completo y sus conexiones quedaban intactas.
                whole = local.copy()
                if CORE_EROSION_PX > 0:
                    whole = distance_transform_edt(whole) > CORE_EROSION_PX
                    if not whole.any():
                        whole = local.copy()
                whole = cut_necks(whole, spacing, typical)
                whole = cut_oversized(whole, spacing, typical)
                sub = skimage.morphology.label(whole)
                if sub.max() == 0:
                    markers[y0:y1, x0:x1][local] = nxt
                    nxt += 1
                    continue
                sub[sub > 0] += nxt - 1
                window = markers[y0:y1, x0:x1]
                window[sub > 0] = sub[sub > 0]
                nxt = int(sub.max()) + 1
                continue
        elif cores.any():
            # Una erosion pequena antes de cortar: adelgaza los cuellos y deja que el
            # corte los encuentre. 'distancia > r' es la erosion por un disco de r.
            if CORE_EROSION_PX > 0:
                eroded = distance_transform_edt(cores) > CORE_EROSION_PX
                if eroded.any():
                    cores = eroded
                del eroded
            # Cortar las uniones, y una segunda pasada sobre lo que siga dando para
            # dos copas. Nada mas.
            #
            # Hubo un paso de relleno de huecos del nucleo y salio peor, en los dos
            # ordenes posibles. Rellenando ANTES de cortar, el relleno tapa el propio
            # cuello que separa dos copas -queda rodeado por las dos- y las suelda:
            # 45 aciertos y 4 fusiones. Rellenando DESPUES, 47 y 2, pero aparecia una
            # copa partida porque el hueco tapado creaba un maximo de distancia falso.
            # Sin rellenar nada: 48 aciertos, 2 fusiones y ninguna copa partida.
            cores = cut_necks(cores, spacing, typical)
            cores = cut_oversized(cores, spacing, typical)
        if not cores.any():
            cores = None
        if cores is None:
            markers[y0:y1, x0:x1][local] = nxt
            nxt += 1
            continue
        sub = skimage.morphology.label(cores)
        sub[sub > 0] += nxt - 1
        window = markers[y0:y1, x0:x1]
        window[sub > 0] = sub[sub > 0]
        nxt += int(sub.max()) - (nxt - 1)
    return markers


def watershed_cut(depthImage, ssMask, crown_separation=0, report=None,
                  pixel_size=0.0):
    """Convierte el mapa de profundidad del modelo en copas.

    Con crown_separation = 0 se conserva el comportamiento historico: se umbraliza
    la profundidad, se erosiona y lo que queda se escribe tal cual. Esa forma
    erosionada ES la copa final, asi que separar y conservar la extension son
    objetivos enfrentados: para partir dos aguajes que se tocan por un frente ancho
    haria falta erosionar tanto que las copas quedarian reducidas a un punto.

    Con crown_separation < 0 el radio se estima solo, por clase y a partir del
    tamano real de las copas de esa ventana (ver estimate_separation), que es lo
    razonable sobre un mosaico grande donde probar valores a mano no es viable.

    Con crown_separation > 0 se hace lo que un Deep Watershed Transform espera: la
    version erosionada se usa solo como MARCADOR, y a partir de ahi un watershed
    reconstruye cada copa hasta el limite de la mascara semantica. Asi las copas se
    separan y ademas recuperan su tamano real. El valor es el radio en pixeles del
    disco con el que se erosionan los marcadores: cuanto mayor, mas agresiva la
    separacion.
    """
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)
    crown_separation = int(crown_separation or 0)
    auto = crown_separation < 0
    # depthImage llega en uint8: negarlo sin convertir daria la vuelta al rango.
    relief = -depthImage.astype(np.int32) if crown_separation != 0 else None

    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)
        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass

        if crown_separation != 0:
            if not ccImage.any():
                continue
            if auto:
                chosen = crown_spacing(semClass, ssMaskClass, pixel_size)
                markers = crown_markers(depthImage, ssMaskClass, chosen, pixel_size)
                local_relief = relief
            else:
                # Modo manual: erosion por un disco. 'distance > r' ES esa erosion,
                # y sale de una sola transformada de distancia.
                chosen = crown_separation
                filled = skimage.morphology.remove_small_holes(ccImage,
                                                               area_threshold=1000)
                markers = skimage.morphology.label(
                    distance_transform_edt(filled) > crown_separation)
                local_relief = relief
            if report is not None:
                report.setdefault(semClass, []).append(chosen)
            if markers.max() == 0:
                continue
            if auto:
                # Los claros de hoja dentro de una copa se tapan aqui, ya con las
                # copas separadas, para no soldar dos vecinas.
                markers = fill_own_holes(
                    markers > 0, CROWN_HOLE_MAX * np.pi * float(chosen) ** 2)
                # Las semillas SON la copa: no se hacen crecer hasta llenar la mascara
                # semantica. La copa queda al tamano que le da el filtrado y el corte,
                # que sobre 56 copas etiquetadas es un diametro mediano de 5,2 m frente
                # a los 5,9 m reales, y cubre el 69 % de la cobertura en vez del 94 %.
                # A cambio, lo que se escribe es lo que el modelo predijo como parte
                # alta de la copa, sin repartir entre semillas el resto de la mancha.
                # La separacion no cambia: 47 aciertos y 2 fusiones con las dos formas.
                resultImage[markers > 0] = csCode
                continue
            grown = watershed(local_relief, markers, mask=ssMaskClass)
            # El watershed deja etiquetas pegadas; se abre un hueco entre ellas para
            # que sigan contandose por separado al reetiquetar el raster final.
            grown[find_boundaries(grown, mode='inner', background=0)] = 0
            resultImage[grown > 0] = csCode
            continue

        ccLabels = skimage.morphology.label(ccImage)
        for ccID in np.unique(ccLabels)[1:]:
            ccIDMask = binary_erosion(ccLabels == ccID, EROSION[semClass])
            resultImage[ccIDMask] = csCode

    return resultImage.astype(np.float32)

def process_instances_raster(raster, crown_dilation=0, semantic=None,
                             pixel_size=0.0):
    """Etiqueta las instancias y, opcionalmente, las encoge o las agranda.

    crown_dilation es el radio en pixeles del disco con el que se ajusta cada copa:
    positivo la dilata, negativo la erosiona hacia su centro. El ajuste va DESPUES de
    etiquetar los componentes conexos, de modo que el recuento no cambia pase lo que
    pase: al dilatar, dos copas contiguas crecen por separado sin fundirse; al
    erosionar, una copa que se quedaria sin un solo pixel conserva su nucleo en vez
    de desaparecer del conteo.

    semantic es la mascara de la segmentacion (1/2/3 por especie). Si se pasa, la
    dilatacion se recorta a ella: una copa no puede crecer hasta donde el modelo dijo
    que no hay palmera. Sin ese recorte las instancias acababan siendo mas grandes
    que la propia cobertura, que es una contradiccion.
    """
    resultImage = np.zeros(shape=raster.shape, dtype=np.float32)
    ninstances = {"mauritia": 0, "euterpe": 0, "oenocarpus": 0}
    crown_dilation = int(crown_dilation or 0)
    selem = skimage.morphology.disk(abs(crown_dilation)) if crown_dilation else None
    erosionar = crown_dilation < 0
    height, width = raster.shape

    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ccImage = (raster == csCode)
        ccImage = skimage.morphology.remove_small_objects(
            ccImage, min_size=min_crown_size(semClass, pixel_size))
        # Aqui habia un remove_small_holes que tapaba los huecos de menos de 1000 px.
        # Se hacia sobre la UNION de todas las instancias de la clase, asi que el hueco
        # de 1 px que watershed_cut abre entre dos copas vecinas, cuando queda rodeado
        # de copa, contaba como hueco pequeno y se rellenaba: volvia a pegar lo que se
        # acababa de separar. La salida del DWT se escribe ahora tal cual sale.
        ccLabels = skimage.morphology.label(ccImage)
        ccIDs = np.unique(ccLabels)[1:]
        ninstances[semClass] = len(ccIDs)

        if selem is None:
            for ccID in ccIDs:
                resultImage[ccLabels == ccID] = csCode
            continue

        # Trabajar sobre el raster entero una vez por instancia seria carisimo: se
        # hace dentro de la caja de cada una, ampliada por el radio del ajuste.
        permitido = None
        if semantic is not None and not erosionar:
            permitido = (semantic == CLASS_TO_SEMANTIC[semClass])

        boxes = find_objects(ccLabels)
        for ccID in ccIDs:
            box = boxes[ccID - 1]
            if box is None:
                continue
            # El margen va en valor absoluto: con un ajuste negativo, restarlo
            # encogeria la caja hasta dejarla vacia en las copas pequenas.
            margen = abs(crown_dilation)
            y0 = max(0, box[0].start - margen)
            y1 = min(height, box[0].stop + margen)
            x0 = max(0, box[1].start - margen)
            x1 = min(width, box[1].stop + margen)
            sub = (ccLabels[y0:y1, x0:x1] == ccID)
            if erosionar:
                ajustada = binary_erosion(sub, structure=selem)
                if not ajustada.any():
                    # La copa no aguanta esa erosion. En vez de perderla del conteo se
                    # deja su punto mas interior, el maximo de la distancia al borde,
                    # que es lo mas parecido al cogollo que queda.
                    distancia = distance_transform_edt(sub)
                    ajustada = distancia >= distancia.max()
                    del distancia
            else:
                ajustada = binary_dilation(sub, structure=selem)
                if permitido is not None:
                    # El recorte limita lo que crece, no lo que ya estaba: la copa
                    # conserva su forma tal como salio del DWT, huecos incluidos, y a
                    # la vez no se extiende a donde el modelo dijo que no hay palmera.
                    ajustada = (ajustada & permitido[y0:y1, x0:x1]) | sub
            resultImage[y0:y1, x0:x1][ajustada] = csCode

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
    # El fondo se escribe como 0 (process_instances_raster parte de np.zeros), no como
    # -9999. Declarar -9999 dejaba el fondo opaco en QGIS; con 0 la capa sale
    # transparente donde no hay copa, como la de cobertura.
    band.SetNoDataValue(0)
    band.WriteArray(mask)
    
    # Configuraciones críticas para visualización en QGIS
    band.SetMetadataItem('AREA_OR_POINT', 'Area')
    try:
        band.ComputeStatistics(False)
    except RuntimeError as err:
        # Con nodata = 0, un raster sin ninguna copa detectada no tiene un solo pixel
        # valido y GDAL no puede muestrear nada. Es un resultado legitimo -en esa zona
        # no hay palmeras- y no debe tumbar la ejecucion entera.
        print(f"Sin píxeles válidos, se omiten las estadísticas: {err}")
    
    out_dataset.FlushCache()
    out_dataset = None
    
    print(f"✓ Raster final guardado: {mask.shape[1]}x{mask.shape[0]}")
    print(f"✓ Geotransform aplicada: {new_gt}")

def apply_instance_onnx(feature_file_list, mask, roi, output_folder, model_path2,
                        window_radius, internal_window_radius, make_tif=True,
                        make_png=False, crown_dilation=0, crown_separation=0,
                        fill_holes=True):
    separation_report = {}
    pixel_size = 0.0
    
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

    if fill_holes:
        # Los huecos internos de la mascara no son solo huecos en la salida: la
        # mascara entra al modelo como cuarto canal Y le pone la imagen a cero fuera
        # de ella, asi que un hueco le borra la imagen al DWT justo ahi. Rellenarlos
        # despues no serviria de nada; tiene que ser antes de inferir.
        clases = datasetresponse.GetRasterBand(1).ReadAsArray()
        rellena = clases.copy()
        anadidos = 0
        for valor in sorted(CLASS_TO_SEMANTIC.values()):
            capa = (clases == valor)
            if not capa.any():
                continue
            lleno = binary_fill_holes(capa)
            nuevos = lleno & ~capa
            rellena[nuevos] = valor
            anadidos += int(nuevos.sum())
        if anadidos:
            memoria = gdal.GetDriverByName('MEM').Create(
                '', datasetresponse.RasterXSize, datasetresponse.RasterYSize, 1,
                datasetresponse.GetRasterBand(1).DataType)
            memoria.SetGeoTransform(datasetresponse.GetGeoTransform())
            memoria.SetProjection(datasetresponse.GetProjection())
            memoria.GetRasterBand(1).WriteArray(rellena)
            datasetresponse = memoria
        print("Huecos rellenados en la mascara antes del DWT: {} px".format(anadidos))
        del clases, rellena

    # Tamano de pixel del raster: con el, la separacion entre copas se fija en metros
    # y deja de depender de lo que mida la mascara en cada ventana.
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        pixel_size = abs(geotransform[1])
    if pixel_size >= MIN_PIXEL_SIZE_M:
        print(f"Pixel size: {pixel_size * 100:.2f} cm")
        for _cls, _m in sorted(MIN_CROWN_SPACING_M.items()):
            print("   {:11s} separacion minima {:.1f} m = {} px".format(
                _cls, _m, max(3, int(round(_m / pixel_size)))))
    else:
        print("Pixel size desconocido o no metrico: la separacion se estimara "
              "del tamano aparente de las copas en cada ventana.")

    bandas = min(dataset.RasterCount, 3)

    output = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype=np.float32) + nodata_value
    # Salida cruda: las dos entradas de watershed_cut, mosaicadas igual que el
    # resultado. Con ellas se puede probar cualquier metodo de separacion en segundos,
    # sin volver a pasar los modelos, que es lo que cuesta los minutos.
    # La profundidad y la mascara se mosaican SIEMPRE, porque la separacion se hace
    # despues sobre el mosaico entero. int8 le llega de sobra a la mascara -sus codigos
    # son -128, -96 y -64- y ahorra tres cuartas partes frente a float32.
    raw_depth = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype=np.uint8)
    raw_ss = np.zeros((dataset.RasterYSize, dataset.RasterXSize), dtype=np.int8)
    print("Mosaico del DWT: {:.0f} MB (profundidad + mascara)".format(
        (raw_depth.nbytes + raw_ss.nbytes) / 1048576))

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
        print(f"PROGRESO fase=instancias hechas={col_idx + 1} de={len(collist)}")
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

        mm = 0
        if internal_window_radius < window_radius:
            mm = int(np.rint(window_radius - internal_window_radius))

        for j, n in enumerate(valid_rows):
            start_row = n - internal_window_radius
            end_row = n + internal_window_radius
            start_col = col - internal_window_radius
            end_col = col + internal_window_radius

            # Asegurar que no nos salimos de los limites
            if (start_row >= 0 and end_row <= output.shape[0] and
                    start_col >= 0 and end_col <= output.shape[1]):
                d = outputBatch[j]
                m = ssMaskBatch[j]
                if mm:
                    d = d[mm:-mm, mm:-mm]
                    m = m[mm:-mm, mm:-mm]
                raw_depth[start_row:end_row, start_col:end_col] = d
                raw_ss[start_row:end_row, start_col:end_col] = m.astype(np.int8)


    # Separar las copas AQUI y no dentro del bucle: cada mancha se juzga entera, y no
    # por el trozo que caia en una ventana. Una mancha grande cortada por el borde
    # parece pequena, y la regla de 'esto es una copa sola' la daba por una sola copa
    # fundiendo las que le quedaban fuera. Medido contra 56 copas etiquetadas a mano:
    # por ventanas 32 aciertos e IoU 0,57; sobre el mosaico 39 aciertos e IoU 0,63.

    if DEBUG_RASTERS:
        # El nombre del raster final aun no esta decidido aqui, asi que los ficheros
        # de depuracion se nombran por la imagen de entrada.
        base = os.path.join(
            output_folder, os.path.splitext(os.path.basename(image_path))[0])
        save_tiff_mask_final(raw_depth.astype(np.float32),
                             base + '_dwt_crudo.tif', dataset)
        save_tiff_mask_final(raw_ss.astype(np.float32),
                             base + '_dwt_crudo_ssmask.tif', dataset)
        save_tiff_mask_final(crown_filter(raw_depth, raw_ss, pixel_size),
                             base + '_dwt_filtrado.tif', dataset)
        print(f"Depuracion: rasteres del DWT junto a {base}")

    print("Separando copas sobre el mosaico completo...")
    try:
        output = watershed_cut(raw_depth, raw_ss, crown_separation,
                               separation_report, pixel_size)
    except Exception as e:
        print(f"Error en watershed cut: {e}")
    if DEBUG_RASTERS:
        save_tiff_mask_final(output.copy(), base + '_dwt_copas.tif', dataset)
    del raw_depth, raw_ss

    # crown_dilation llega en metros: en pixeles el numero no significa nada, porque
    # 5 px son 22 cm a 4,5 cm/px y 10 cm a 2 cm/px.
    if crown_dilation and pixel_size and pixel_size >= MIN_PIXEL_SIZE_M:
        dilation_px = int(round(crown_dilation / pixel_size))
        verbo = "erosion" if dilation_px < 0 else "dilatacion"
        print(f"Crown adjustment: {crown_dilation} m = {dilation_px} px ({verbo})")
    else:
        dilation_px = int(round(crown_dilation))
        if crown_dilation:
            print(f"Crown dilation: {dilation_px} px (sin tamaño de píxel conocido)")

    if separation_report:
        unit = ("minimum crown spacing, px" if crown_separation < 0
                else "erosion radius, px")
        print(f"Crown separation actually used ({unit}):")
        for semClass in sorted(separation_report):
            values = separation_report[semClass]
            print("   {:11s} median {:.0f} (min {:.0f}, max {:.0f}, {} windows)".format(
                semClass, float(np.median(values)), min(values), max(values), len(values)))

    # La mascara semantica completa: 1 byte por pixel, y evita que la dilatacion se
    # salga de donde el modelo dijo que hay palmera.
    semantic_full = None
    if dilation_px > 0:
        semantic_full = datasetresponse.GetRasterBand(1).ReadAsArray()

    output, quantification = process_instances_raster(output, dilation_px,
                                                     semantic_full, pixel_size)
    del semantic_full

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