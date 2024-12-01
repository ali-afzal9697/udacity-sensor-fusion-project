# Final Project Report

## 1. Recap of the Four Tracking Steps

### Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is used to estimate the state of a moving object over time. It extends the basic Kalman
Filter to handle non-linear models. In this project, the EKF was implemented to predict and update the state of tracked
objects using lidar and camera measurements.

### Track Management

Track management involves creating, updating, and deleting tracks of objects. In this project, tracks were initiated
when new objects were detected by LiDAR, updated with new measurements, and deleted if they were not detected for a
certain number of frames. This ensures that only relevant and confirmed tracks are maintained.

### Data Association

Data association is the process of matching measurements to existing tracks. In this project, the Single Nearest
Neighbor approach was used to associate measurements with tracks based on the Mahalanobis distance. This helps in
accurately updating the state of the tracked objects.

### Camera-Lidar Sensor Fusion

Sensor fusion combines data from multiple sensors to improve the accuracy of object tracking. In this project, camera
and lidar data were fused to leverage the strengths of both sensors. Lidar provides accurate distance measurements,
while the camera provides rich visual information.

### Results

The results highly depends on the quality of the object detection models. If one of sensors doesn't provide good
detection performance, the tracking performance will be affected significantly. Also finding the right set of tuning
parameters for the filters is challenging.

### Most Difficult Part

I believe the most difficult part of the project was to derive the kalman filter equations to predict and update the car
state. Another thing that is very related to that is tuning the filter parameters which need a very good understanding
of the
system.

## 2. Benefits of Camera-Lidar Fusion

### Theoretical Benefits

- **Redundancy**: Fusion provides redundancy, which increases the reliability of the system.
- **Complementary Information**: Lidar provides precise distance measurements, while the camera provides detailed visual
  information. Combining these can improve the overall accuracy.

### Limitations

The current fusion system heavily relies on the performance of LiDAR detection so if the lidar doesn't detect the
object,
the object will be lost in the tracking process. Having said that, camera detection can improve the detection accuracy
but it cannot help with detecting the objects that is
not detected by LiDAR sensor.

## 3. Challenges in Real-Life Scenarios

### Sensor Calibration

Sensor calibration is crucial for accurate fusion. Misalignment or calibration errors can lead to incorrect
measurements.

### Environmental Factors

Real-life scenarios involve varying lighting conditions, weather, and dynamic environments, which can affect sensor
performance.

### Computational Complexity

Fusion algorithms can be computationally intensive, requiring efficient implementation to run in real-time.

## 4. Future Improvements

### Improved Data Association

As mentioned during the course, there are more advanced association techniques that can be used to improve the
association task.

### Enhanced Sensor Fusion

We can use better sensor fusion technique to take advantage of camera data even though the object is not detected by
LiDAR.

### Machine Learning

Fine tune the object detection models to improve on the Waymo dataset. Currently, pre-trained LiDAR object detection has
been used.

### Real-Time Optimization

Some real-time optimization techniques can be used to improve the performance of the tracking system in order to be used
in the real life scenarios.