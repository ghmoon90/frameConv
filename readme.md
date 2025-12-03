# Frame Conversion Operator

## User Manual & Processing Guide

## 1. Introduction

The Frame Conversion Operator is a Python-based GUI application designed to combine multiple sequential frames—sourced from videos, animated GIFs, or image sequences—into a single, static image. This process, often called computational photography, allows you to create specialized effects like long exposures and object removal.

The application supports four different processing modes to achieve distinct visual results.

## 2. Requirements & Setup

Before running the application, ensure you have the following installed:

* **Python:** The application requires Python 3.x installed on your system.

* **Libraries:** You must have the following libraries installed:

```
pip install opencv-python numpy pillow
```
or install with requirements.txt
```
pip install -r requirements.txt
```


* **GUI:** The application uses `tkinter`, which is typically included with standard Python installations.

## 3. Interface Overview

### Input Selection

Use **Browse** to select one file (Video/GIF) or multiple image files (`.jpg`, `.png`, etc.). The application automatically detects the input type.

### Output Image

Use **Save As** to specify the file path and format (e.g., `result.png`) for the final processed image.

### Frame Skip

A value greater than 1 tells the program to skip frames (e.g., skip=5 means only process 1 out of every 5 frames). Higher values result in **faster processing** but may slightly **reduce quality**.

## 4. Processing Modes Explained

### Input Video and GIF
- Nuriho Rocket Launching (27 Nov, 2025)
	<video width="400" height="300" controls>
		<source src="ex/Nuri.MOV#t=12"  type="video/mp4">
	</video>
- Car Crossing (from Google Image) 
 <img src = "ex/carcross.gif">



### Average

* **Description:** Calculates the mathematical **average** of the color value for every pixel across all processed frames.

* **Effect:** Standard long exposure; creates a silky, blurred effect on moving objects (waterfalls, clouds). 

* **Memory:** Low consumption (processes frames incrementally).

* results 
 <img src = "ex/Avg.png">


### Brightest

* **Description:** For each pixel location, it keeps the **highest (brightest)** color value recorded across all frames.

* **Effect:** Captures light trails (cars, stars, fireworks) against a dark background. 

* **Memory:** Low consumption (processes frames incrementally).

* results 
 <img src = "ex/Brightest.png">

### Darkest

* **Description:** For each pixel location, it keeps the **lowest (darkest)** color value recorded across all frames.

* **Effect:** Useful for removing bright, temporary flashes of light. Can deepen shadows.

* **Memory:** Low consumption (processes frames incrementally).

* results 
 <img src = "ex/darkest.png">


### Median (Memory-Safe)

* **Description:** Finds the middle pixel value (the median) in the sequence for every location.

* **Effect:** Excellent for removing moving objects (people, cars) from a static scene, as they are not present in the median frame. Known as **"Ghost Removal"**. 

* **Memory:** **Memory-Safe Batching.** The code reads the source multiple times, processing the image in small horizontal strips. This is slower than the other modes but prevents high RAM consumption for long videos.

* results 
 <img src = "ex/Median.png">

*Frame Conversion Operator | Version 1.0*


