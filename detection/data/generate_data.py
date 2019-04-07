from arcgis.gis import GIS
from arcgis.features import SpatialDataFrame
from arcgis.raster import ImageryLayer
from arcgis.geometry import Polygon
from arcgis.geometry import Geometry
import sys
import json
import arcgis_config

# type of coordinate referrence system 
crs_id = 3857
gis = GIS("https://www.arcgis.com", arcgis_config.username, arcgis_config.password)

shp_file = 'raw/bottom_part.shp'
building_data = SpatialDataFrame.from_featureclass(shp_file)
# print(type(building_data.geometry))
# print(df.dtypes)
# print(df.shape)

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


# find the min/max of x and y coordinates of all buildings. 
# min_x = -8791910.9147
# min_y = 4234675.8780
# max_x = -8711883.4966
# max_y = 4311092.4135
min_x = -8784796.6706
min_y = 4242405.9333
max_x = -8752365.2401
max_y = 4261174.7474
# we operate on 200 level
tile_size = 200
image_size = 224
x_num_tile = int((max_x - min_x) / tile_size) + 1
y_num_tile = int((max_y - min_y) / tile_size) + 1

def overlap(a, b):
    x1, y1, x2, y2 = a
    def point_in(p, box):
        m1, n1, m2, n2 = box
        x, y = p
        if x >= m1 and x <= m2 and y >= n1 and y <= n2:
            return True
        return False
    return (point_in((x1, y1), b)) or (point_in((x2, y1), b)) or (point_in((x2, y2), b)) or (point_in((x1, y2), b))

# iterate through image tiles on naip_image_layer starting from bottom row 
for y_idx in range(y_num_tile):
    for x_idx in range(x_num_tile):
        image_name = str(y_idx*x_num_tile + x_idx)
        x_start = min_x + x_idx * tile_size
        y_start = min_y + y_idx * tile_size

        # export annotations for buildings in the image 
        tile_image_geometry = Geometry({
            "rings" : [[[x_start,y_start],[x_start+tile_size,y_start],[x_start+tile_size,y_start+tile_size],[x_start,y_start+tile_size]]],
            "spatialReference" : {"wkid" : crs_id}
        })
        annotations = []
        for idx, bbox in enumerate(building_data.geometry):
            # bbox is polygon
            bbox_extent = bbox.extent
            tile_extent = tile_image_geometry.extent
            try:
                # if bbox overlaps with tile image extent, record the building bbox
                if overlap(bbox_extent, tile_extent):
                    # bbox contains normalized [xywh]
                    x1_r, y1_r, x2_r, y2_r = bbox_extent
                    # clipping
                    x1 = max(x_start, x1_r)
                    y1 = max(y_start, y1_r)
                    x2 = min(x2_r, x_start+tile_size)
                    y2 = min(y2_r, y_start+tile_size)
                    x = (x1-x_start) / tile_size
                    y = (tile_size+y_start-y2) / tile_size
                    w = (x2-x1) / tile_size
                    h = (y2-y1) / tile_size
                    # keep big building instance 
                    if not ((x2_r - x1_r) * (y2_r - y1_r)) >= (0.3 * tile_size * tile_size):
                        # else, drop significantly clipped building instances 
                        if ((x2 - x1) * (y2 - y1)) <= (0.5 * ((x2_r - x1_r) * (y2_r - y1_r))):
                            continue
                    assert x >= 0 and x <= 1
                    assert y >= 0 and y <= 1
                    assert w >= 0 and w <= 1
                    assert h >= 0 and h <= 1
                    building_type = int(building_data['OCCUP_TYPE'][idx])
                    # print(building_type)
                    # 0 is residential and 1 is non-residential
                    building_class = 0 if building_type in res_types else 1
                    annotations.append({
                        'bbox': [x, y, w, h],
                        'class': building_class
                    })
            except:
                continue
        if len(annotations) == 0:
            continue
        annotation_output = './tmp_annotations/'+image_name+'.json'
        with open(annotation_output, 'w') as f:
            json.dump(annotations, f, indent=4)
        print('annotation outputted to %s' % annotation_output)

        # export the tile image
        extent = {'xmin':x_start, 'ymin':y_start, 'xmax':x_start + tile_size, 'ymax':y_start + tile_size, 'spatialReference':crs_id}
        naip_image_layer.export_image(
            extent, 
            size=[image_size, image_size], 
            save_folder='./tmp_images',
            save_file=image_name+'.jpg',
            f='image', 
            export_format='jpg', 
            adjust_aspect_ratio=False
        )
        print('image outputted to %s' % ('./images/'+image_name+'.jpg'))
        