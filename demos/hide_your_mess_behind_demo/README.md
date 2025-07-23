# Hide Your Mess Behind

## Description

This demo demonstrates how to use the OpenVINO toolkit in NodeJS to blur the background of video. 

There are 2 possible ways to run the demo - using executable or source code in NodeJS.

## Running the demo using executable file

### Installers

Download installers of the compiled app. They are available for Windows and Linux.

| OS | Installer |
|---|---|
| Linux | [DEB](https://github.com/openvinotoolkit/openvino_build_deploy/releases/download/hide_your_mess_behind_v1.1/hide-your-mess-behind_1.1.0_amd64.deb) [RPM](https://github.com/openvinotoolkit/openvino_build_deploy/releases/download/hide_your_mess_behind_v1.1/hide-your-mess-behind-1.1.0.x86_64.rpm) |
| Windows | [EXE](https://github.com/openvinotoolkit/openvino_build_deploy/releases/download/hide_your_mess_behind_v1.1/hide-your-mess-behind.Setup.1.1.0.exe) |

#### Windows

Double-click the installer and follow the instructions. Then run the app from the Start menu.

#### Ubuntu

```bash
sudo dpkg -i hide-your-mess-behind_1.1.0_amd64.deb
hide-your-mess-behind
```

## Running the demo using source code and NodeJS

### Requirements

Ensure that you have Node.js (with npm) installed on your system. The app was developed and tested using *node v20.15.0* and *npm 10.8.2*.

### Getting started

Before running the app you have to initialize the electron project and install the required packages. Do it by running the following commands in the app folder:

```bash
npm init -y
npm install
```

### Running the demo

Once you've completed the initial setup, you can start the app anytime by running the following command in the app folder:

```bash
npm start
```

## Using the Demo

### Turn on the video

When you open the app, the following view will appear:

![image](https://github.com/user-attachments/assets/b9852e1e-3fa7-4375-afb9-8976cd9cf325)

Select the chosen video source from the control panel. Then click _Start_ button to start the streaming.

![image](https://github.com/user-attachments/assets/cd5a86e2-8865-4736-93e6-e2e0eb9b37f2)

Later you can turn off streaming by clicking _Stop_ button.


### Turn on the inference

To turn on blurring you have to turn on inference using the _Inference_ switch. Below it, you can notice a panel, where you can choose the inference device (e.g. AUTO, GPU, CPU, NPU). 

![image](https://github.com/user-attachments/assets/e6925e6b-0d81-41da-b9b0-c4f21f173681)

You can change the inference device or video source, and turn on and off inference, and streaming anytime.

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=demos/hide_your_mess_behind_demo&file=README.md" />
