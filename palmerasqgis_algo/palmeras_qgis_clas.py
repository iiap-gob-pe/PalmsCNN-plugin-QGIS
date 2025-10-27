
import os
from qgis.core import (QgsRasterLayer,QgsVectorLayer,
                       QgsFeatureRequest, QgsVectorDataProvider,
                       QgsField,)
from qgis import processing
from qgis.PyQt.QtCore import QVariant



def apply_toolsqgis(OUTPUT_RASTER,name_mask_clas,mau,eut,oeno):

    OUTPUT_VEC = os.path.join(name_mask_clas.split('.')[0] + '_poly.shp')
    #OUTPUT_CEN = os.path.join(OUTPUT_RASTER.split('.')[0] + '_centroides.shp')
    #OUTPUT_CSV = os.path.join(OUTPUT_RASTER.split('.')[0] + '_atributos.csv')
    OUTPUT_REPORT = os.path.join(OUTPUT_RASTER.split('.')[0] + '_reporte.csv')

    raster_source = QgsRasterLayer(name_mask_clas)
    #Poligonize
    res = processing.run("gdal:polygonize",
        {'INPUT' : raster_source,
        'BAND' : 1,
        'FIELD' : 'ID',
        'OUTPUT': OUTPUT_VEC})
    vec = res['OUTPUT']##string
    #Repair Shapefile
    res2 = processing.run("qgis:repairshapefile",
        {'INPUT': vec,
        'OUTPUT': vec})
    vec2 = res2['OUTPUT']##string
    vlayer = QgsVectorLayer(vec2, "layerpalms","ogr")
    caps = vlayer.dataProvider().capabilities()
    feats= vlayer.getFeatures()
    dfeats =[]
    #Remove features=0
    if caps & QgsVectorDataProvider.DeleteFeatures:
        for feat in feats:
            if feat['ID'] == 0 :
                dfeats.append(feat.id())
        res = vlayer.dataProvider().deleteFeatures(dfeats)
    #Add  Fields
    if caps & QgsVectorDataProvider.AddAttributes:
        res = vlayer.dataProvider().addAttributes([QgsField('CLASE', QVariant.Int),
                                                   QgsField('ESPECIE', QVariant.String),
                                                   QgsField('ÁREA(m2)', QVariant.Double),
                                                   QgsField('UTM(ESTE)', QVariant.Double),
                                                   QgsField('UTM(NORTE)', QVariant.Double),])
        vlayer.updateFields()
    fc = vlayer.featureCount()  
    c1=0 #count each palm
    c2=0
    c3=0
    ca1=0 #sum all the area
    ca2=0
    ca3=0
    if caps & QgsVectorDataProvider.ChangeAttributeValues:
        for i in range(0, fc):
            feat = vlayer.getFeature(i)
            geom = feat.geometry()
            if (feat['ID'] == 1) : #0=FIELD ID
                especie = {2 : 'Mauritia flexuosa'} #2= FIELD ESPECIE
                #c1 = c1+1
                ca1 = ca1 + geom.area()
            elif (feat['ID'] == 2):
                especie = {2 : 'Euterpe precautoria'}
                #c2 = c2+1
                ca2 = ca2 + geom.area()
            elif(feat['ID'] == 3):
                especie = {2 : 'Oenocarpus bataua'}
                #c3 = c3+1
                ca3 = ca3 + geom.area()
            res=vlayer.dataProvider().changeAttributeValues({feat.id():especie})
            c1 = mau
            c2 = eut
            c3 = oeno
            areacopa={3 : geom.area()}  #3= FIELD ESPECIE
            cx = {4 : geom.centroid().asPoint().x()} #4= FIELD UTMESTE
            cy = {5 : geom.centroid().asPoint().y()} #5= FIELD UTMNORTE
            clase = {1 : feat['ID']} #1= CLASE
            idchange = {0 : feat.id()} #0= FIELD ID
            res=vlayer.dataProvider().changeAttributeValues({feat.id():areacopa})
            res=vlayer.dataProvider().changeAttributeValues({feat.id():cx })
            res=vlayer.dataProvider().changeAttributeValues({feat.id():cy })
            res=vlayer.dataProvider().changeAttributeValues({feat.id():clase })
            res=vlayer.dataProvider().changeAttributeValues({feat.id():idchange })
    
    
    fieldnames = [field.name() for field in vlayer.fields()]
    features = vlayer.getFeatures()
    #with open(OUTPUT_CSV, 'w') as output_file:
        #line = ','.join(name for name in fieldnames) + '\n'
        #output_file.write(line)
        #for current, f in enumerate(features):
            #line = ','.join(str(f[name]) for name in fieldnames) + '\n'
            #output_file.write(line)

    labelnames = ['ESPECIE', 'CANTIDAD DE INDIVIDUOS', 'AREA TOTAL(ha)' ]
    rowname1 = ['Mauritia flexuosa', c1, ca1/10000 ]
    rowname2 = ['Euterpe precautoria', c2, ca2/10000]
    rowname3 = ['Oenocarpus bataua', c3, ca3/10000]
    with open(OUTPUT_REPORT, 'w') as output_file:
        line = ','.join(name for name in labelnames) + '\n'
        output_file.write(line)
        line = ','.join(str(name) for name in rowname2) + '\n'        
        output_file.write(line)
        line = ','.join(str(name)for name in rowname1) + '\n'        
        output_file.write(line)
        line = ','.join(str(name) for name in rowname3) + '\n'        
        output_file.write(line)

    #res = processing.run("native:centroids",
    #    {'INPUT' : vlayer ,
    #    'ALL_PARTS' : True,
     #   'OUTPUT':  OUTPUT_CEN })
    del(vlayer)
    if c1==0:
        ca1=0
    elif c2==0:
        ca2=0
    elif c3==0:
        ca3=0    
    return ca1/10000,ca2/10000, ca3/10000, OUTPUT_REPORT# c1, c2, c3, ca1/10000,ca2/10000, ca3/10000,OUTPUT_VEC,OUTPUT_CEN,OUTPUT_CSV,OUTPUT_REPORT
    #print(count1) 
    #print(count2)
    #print(count3)
    #print('Area Mauritia: ', counta1/10000)
    #print('Area Euterpe: ', counta2/10000)
    #print('Area Oenocarpus: ', counta3/10000)
    """
    
#processing.runalg("gdalogr:polygonize",OUTPUT_RASTER,"DN",None)
    raster_source=QgsRasterLayer(OUTPUT_RASTER)
#processing.algorithmHelp("gdal:polygonize")
    res= processing.run("gdal:polygonize",
        {'INPUT':raster_source,
        'BAND':1,
        'FIELD': 'CLASS_ID',
        'OUTPUT':OUTPUT_VEC})

    vec= res['OUTPUT']##string
    res2=processing.run("qgis:repairshapefile",
        {'INPUT':vec,
        'OUTPUT':vec})
    vec2= res2['OUTPUT']##string
    vlayer = QgsVectorLayer(vec2, "layerpalms","ogr")
    if not vlayer:
        print("Layer failed to load!")

    caps = vlayer.dataProvider().capabilities()
    feats= vlayer.getFeatures()
    dfeats =[]

    if caps & QgsVectorDataProvider.DeleteFeatures:
        for feat in feats:
            if feat['CLASS_ID'] == 0 :
                dfeats.append(feat.id())
        res = vlayer.dataProvider().deleteFeatures(dfeats)

    """
    """with edit(vlayer):z
        request = QgsFeatureRequest().setFilterExpression('"CLASS_ID" = 0')
        request.setSubsetOfAttributes([])
        request.setFlags(QgsFeatureRequest.NoGeometry)
        for f in vlayer.getFeatures(request):
            vlayer.deleteFeature(f.id())
     """   
    