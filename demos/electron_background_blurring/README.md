# Hide Your Mess Behind

## Description

This demo demonstrates how to use OpenVINO framework in NodeJS to blur the background of video. 

## Requirements

Ensure that you have Node.js (with npm) installed on your system. The app was developed and tested using *node v20.15.0* and *npm 10.8.2*.

## Getting started

Before running the app you have to initialize electron project and install required packages. Do it by running following commands in the app folder:

```bash
npm init -y
npm install
```

## Running the Demo

Once you've completed the initial setup, you can start the app anytime by running the following command in the app folder:

```bash
npm start
```

### Turn on the video

When you open the app, following view will appear

<img width="584" alt="video-stopped" src="https://github.com/user-attachments/assets/33d9ab98-40e9-4fc3-b9e3-b302aa49bc9d">

Select the chosen video source from control panel. Then click _Start_ button to start the streaming.

<img width="587" alt="inference-off" src="https://github.com/user-attachments/assets/ba8f4b6f-33a1-43cf-8885-265f117f482e">

Later you can turn off streaming by clicking _Stop_ button.


### Turn on the inference

To turn on blurring you have to turn on inference using the _Inference_ switch. Below it you can notice panel, where you can choose inference device (e.g. AUTO, GPU, CPU, NPU). 

<img width="586" alt="inference-on" src="https://github.com/user-attachments/assets/eb9e1b75-8efb-4d6d-80e1-a9c438048bf7">

You can change inference device or video source, turn on and off inference and streaming anytime.
