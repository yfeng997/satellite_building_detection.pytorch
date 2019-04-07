from arcgis.gis import GIS
from arcgis.features import SpatialDataFrame
from arcgis.raster import ImageryLayer
from arcgis.geometry import Polygon
from arcgis.geometry import Geometry
import sys
import json
import os 

# type of coordinate referrence system 
crs_id = 3857
gis = GIS("https://www.arcgis.com", "YuansongFengPro", "Fys19970807!")

shp_file = 'raw/bottom_part.shp'
building_data = SpatialDataFrame.from_featureclass(shp_file)
output_dir = './images'

# naip = gis.content.search('Views', 'Imagery Layer', outside_org=True)
naip = gis.content.get('3f8d2d3828f24c00ae279db4af26d566')
# for layer in naip.layers:
#     print(layer)
naip_image_layer = naip.layers[0]
# naip_image_layer = apply(naip_image_layer, 'FalseColorComposite')
# print(naip_image_layer.extent)

# redefine occupancy type to be residential(1) and non-residential(2)
with open('residential_occupancy_types.json', 'r') as f:
    res_types = json.load(f)

image_size = 224
res_idx = 0
nonres_idx = 0
# number of output class for each category
output_num = 10000

# iterate through building instances on image layer and crop down each instance 
for idx, bbox in enumerate(building_data.geometry):
    try:
        building_type = int(building_data['OCCUP_TYPE'][idx])
        if building_type in res_types:
            continue
            # res_idx += 1
            # print(res_idx)
            # if res_idx >= output_num:
            #     continue
            # output_folder = os.path.join(output_dir, 'residential')
            # output_filename = str(res_idx)+'.jpg'
        else:
            nonres_idx += 1
            if nonres_idx >= output_num:
                break
            output_folder = os.path.join(output_dir, 'non_residential')
            output_filename = str(nonres_idx)+'.jpg'
        # bbox is polygon
        x1_r, y1_r, x2_r, y2_r = bbox.extent
        pad = (x2_r-x1_r) / 4
        extent = {'xmin':x1_r-pad, 'ymin':y1_r-pad, 'xmax':x2_r+pad, 'ymax':y2_r+pad, 'spatialReference':crs_id}
        naip_image_layer.export_image(
            extent, 
            size=[image_size, image_size], 
            save_folder=output_folder,
            save_file=output_filename,
            f='image', 
            export_format='jpg', 
            adjust_aspect_ratio=False
        )
        print('image outputted to %s' % (output_folder+'/'+output_filename))
        if (res_idx == output_num) and (nonres_idx == output_num):
            break
    except Exception as e:
        print(e)
        continue



