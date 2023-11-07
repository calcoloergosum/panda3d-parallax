# Parallax with Head Tracking

Parallax is an illusion of depth.

## Sample

![samples/sample.gif]
Created with `python main.py samples/data/ship.egg --distance 500 --model-hpr "(90, 0, 0)"`

## References

- 1977 [GENERAL CLIPPING ON AN OBLIQUE VIEWING FRUSTRUM](https://dl.acm.org/doi/pdf/10.1145/563858.563898)
- 1992 [The CAVE: audio visual experience automatic virtual environment](https://doi.org/10.1145%2F129888.129892)
- 2008 [Johnny Lee](https://www.youtube.com/watch?v=Jd3-eiid-Uw)
- 2014 [Amazon Firephone](https://en.wikipedia.org/wiki/Fire_Phone)
- [Daito Manabe](https://daito.ws/en/)

## Run

1. Make a scene for panda3d to render. [Converting from Blender](https://docs.panda3d.org/1.10/python/tools/model-export/converting-from-blender).
2. run using `python main.py ./samples/data/ship.egg`

## TODO

- [x] Single web camera
- [ ] Intel realsense (Linux/Windows only)
- [ ] StereoLabs - ZED series
- [ ] Poorly calibrated multiple web cameras
- [ ] Easily configurable HDRI
- [ ] Light Field Display (Output as plenoptic function)
