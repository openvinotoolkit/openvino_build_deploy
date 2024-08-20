const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');
const path = require('path');

module.exports = { detectDevices, runModel, takeTime }

const core = new ov.Core();
const ovModels = new Map();
let mat = null;
let resizedMat = null;
let paddedImg = null;

const model = core.readModel(path.join(__dirname, "../models/selfie_multiclass_256x256.xml"));

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function preprocessMat(image, targetHeight = 256, targetWidth = 256) {
    // rows == height
    // cols == width

    // RESIZING
    if (image.rows < image.cols){
        const height = Math.floor(image.rows / (image.cols / targetWidth));
        if (resizedMat == null || resizedMat.size().width != targetWidth || resizedMat.size().height != height){
            resizedMat = new cv.Mat(height, targetWidth, cv.CV_8UC3);
        }
        console.log(resizedMat.size().height);
        cv.resize(image, resizedMat, resizedMat.size());
    } else {
        const width = Math.floor(image.cols / (image.rows / targetHeight));
        if (resizedMat == null || resizedMat.size().width != width || resizedMat.size().height != targetHeight){
            resizedMat = new cv.Mat(targetHeight, width, cv.CV_8UC3);
        }
        console.log(resizedMat.size().height);
        cv.resize(image, resizedMat, resizedMat.size());
    }

    //CHANGING FROM 4-CHANNEL BGRA TO 3-CHANNEL RGB
    cv.cvtColor(resizedMat, resizedMat, cv.COLOR_BGRA2RGB);

    // PADDING
    const rightPadding = Math.max(0,targetWidth - image.cols);
    const bottomPadding = Math.max(0,targetHeight - image.rows);

    if (paddedImg == null){
        paddedImg = new cv.Mat(targetHeight,targetWidth,cv.CV_8UC3);
    }
    
    cv.copyMakeBorder(
        resizedMat,
        paddedImg,
        0,
        bottomPadding,
        0,
        rightPadding,
        cv.BORDER_CONSTANT,
        [0,0,0,0]    
    );

    return {
        image : paddedImg,
        paddingInfo : { bottomPadding, rightPadding }
    };
}


async function runModel(img, width, height, device){
    // if device in ovModels, use precompiled model, otherwise load and compile model and ut to the map

    // CONVERTION TO MAT:

    if (mat == null || mat.data.length != img.data.length){
        mat = new cv.Mat(height, width, cv.CV_8UC4);
    }
    mat.data.set(img.data);

    // PREPROCESSING:
    let processedImage = preprocessMat(mat);

    console.log(mat.data.length, img.data.length);

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