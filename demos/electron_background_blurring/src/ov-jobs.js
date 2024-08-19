const { addon: ov } = require('openvino-node');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');
const path = require('path');
const { Buffer } = require('buffer');
const { getImageData } = require('./helpers.js');

module.exports = { detectDevices, runModel, takeTime }

const core = new ov.Core();
const ovModels = new Map();

const model = core.readModel(path.join(__dirname, "../models/selfie_multiclass_256x256.xml"));

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function resizeAndPad(image, targetHeight = 256, targetWidth = 256) {
    let height = image.rows;
    let width = image.cols;

    let newWidth, newHeight;
    if (height < width) {
        newWidth = targetWidth;
        newHeight = Math.floor(height / (width / targetWidth));
    } else {
        newHeight = targetHeight;
        newWidth = Math.floor(width / (height / targetHeight));
    }

    let resizedImg = new cv.Mat();
    let newSize = new cv.Size(newWidth, newHeight);
    cv.resize(image, resizedImg, newSize, 0, 0, cv.INTER_LINEAR);

    let rightPadding = targetWidth - newWidth;
    let bottomPadding = targetHeight - newHeight;

    let paddedImg = new cv.Mat();
    let black = new cv.Scalar(0, 0, 0, 255);
    cv.copyMakeBorder(resizedImg, paddedImg, 0, bottomPadding, 0, rightPadding, cv.BORDER_CONSTANT, black);

    resizedImg.delete();

    return { paddedImg: paddedImg, paddingInfo: { bottomPadding: bottomPadding, rightPadding: rightPadding } };
}


async function runModel(img, width, height, device){
    // if device in ovModels, use precompiled model, otherwise load and compile model and ut to the map

    // CONVERTION TO MAT:
    // const mat = new cv.Mat(height, width, cv.CV_8UC4);
    // mat.data.set(img.data);


    const startTime = performance.now();
    // INFERENCE OpenVINO (TODO)
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;
    return {
        img : img, 
        inferenceTime : inferenceTime.toFixed(2).toString()
    };
}


function takeTime(){
    return performance.now();
}