# Phenotypic Trait Extraction Code



## PointNeXt-MS: An advanced approach for phenotyping individual tree sapling stem and leaf 3D structural from LiDAR point clouds by field phenotyping platform
## Environment Configuration
* Python3.9
* Pytorch1.10 or later
* GPU training is highly recommended
* See `requirements.txt` for detailed environment configuration

## File Structure:
```
    ├── Ginkgo Phenotype/                      
    │   ├── normal vector/                                        # Store point cloud data with normal vectors
    │   ├── phenotype Extraction/                                 # Phenotypic Trait Extraction Algorithm
    │   │   ├── Branch Inclination Angle Extraction.py              # Branch Angle Extraction
    │   │   ├── Leaf Area Extraction.py                             # Leaf Area Extraction
    │   │   └── Stem Straightness Extraction.py                     # Stem Straightness Extraction
    │   ├── pth/                                                  # Store the best model
    │   │   └── best.pth   
    │   ├── result/                                               # result
    │   ├── Test data/                                            # Test data                      
    │   ├── Leaf Segmentation.py                                  # Leaf Segmentation
    │   ├── model.py                                              # model
    │   ├── Point Cloud Visualization.py                          # Point Cloud Visualization
    │   ├── predict.py                                            # Prediction Code
    │   ├── Trunk Fitting.py                                      # Trunk Fitting
    └── └──main.py                                                # main
```

- When using (the program/code), please select and run the corresponding code.

```
  Prediction Code Script： python predict.py --input_file tree8_normals.txt --model_path outputs/best_model.pth --num_points 2048 --output_file outputs/pred.txt --use_gpu
```
