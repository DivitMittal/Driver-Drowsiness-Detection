<h1 align='center'>Driver-Drowsiness-Detection</h1>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Demo](#demo)

## Introduction

Driver drowsiness is a major contributor to road accidents. This **Driver Drowsiness Detection** system uses **Siamese Neural Networks** and computer vision techniques to monitor a driver's facial features in real-time and detect signs of fatigue. By analyzing eye states, head pose, and mouth movements (including yawning), the system identifies potential drowsiness and enhances road safety.

## Features

- **Real-Time Drowsiness Detection:** Monitors the driver's face continuously for signs of fatigue.
- **Eye Aspect Ratio (EAR):** Utilizes EAR to determine prolonged eye closures, a common sign of drowsiness.
- **Head Pose Estimation:** Detects abnormal head movements such as nodding or tilting.
- **Mouth and Yawning Detection:** Analyzes mouth movements using computer vision techniques to identify yawning, another common fatigue indicator.
- **Siamese Neural Networks:** All detection (eye, head pose, and yawning) is powered by a Siamese Neural Network, which allows the system to effectively compare facial features and detect signs of drowsiness.

## Technologies Used

- **Python 3.8+**
- **Tensorflow & Keras (Siamese Neural Networks)** for feature comparison and detection.
- **OpenCV** for real-time computer vision tasks.
- **Dlib** for facial landmark detection.
- **Imutils** for image processing utilities.