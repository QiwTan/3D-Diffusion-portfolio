# 3D-Diffusion

## Project Overview

The main goal of this project is to predict the 3D pressure distribution in space around a drone during its flight.

Specifically, the project takes a (128x128x64) 3D tensor as input, representing the shape and position of the drone in 3D space, and outputs a (128x128x64) 3D tensor indicating the pressure distribution in the surrounding 3D space.

<div align="center">
  <h3>Input: Labels (Shape and Position of the Drone)</h3>
  <img src="voxel_labels_visualization.gif" alt="Input Labels" width="45%" />
</div>

<div align="center">
  <h3>Ground Truth and Prediction</h3>
  <img src="voxel_visualization_ground_truth.gif" alt="Ground Truth" width="45%" />
  <img src="voxel_visualization_25000.gif" alt="Prediction" width="45%" />
</div>
