# Building detection
Detect buildings from satellite images and perform classification. Region proposals and classification are done in one scan following the model structure of YOLO v3. 

## Data Preparation
Images containing multiple buildings along with annotations are generated with a python script inside data folder. Note that the current workflow requires a Pro account in ArcGis and a shapefile for the interested region. Following steps are taken:
1. Log into ArcGis
2. Access the satellite image layer on ArcGis and generate image tiles of the interested region
3. For each image tile, iterate through all building features in the shapefile and create an annotation file


## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
