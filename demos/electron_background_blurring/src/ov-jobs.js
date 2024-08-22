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

let model = null;

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
        cv.resize(image, resizedMat, resizedMat.size());
    } else {
        const width = Math.floor(image.cols / (image.rows / targetHeight));
        if (resizedMat == null || resizedMat.size().width != width || resizedMat.size().height != targetHeight){
            resizedMat = new cv.Mat(targetHeight, width, cv.CV_8UC3);
        }
        cv.resize(image, resizedMat, resizedMat.size());
    }

    //CHANGING FROM 4-CHANNEL BGRA TO 3-CHANNEL RGB
    cv.cvtColor(resizedMat, resizedMat, cv.COLOR_BGRA2RGB);

    // PADDING
    const rightPadding = Math.max(0,targetWidth - resizedMat.cols);
    const bottomPadding = Math.max(0,targetHeight - resizedMat.rows);

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

async function createInferRequest(device){ 
    let compiledModel = null;
    if (model == null){
        model = await core.readModel(path.join(__dirname, "../models/selfie_multiclass_256x256.xml"));
        console.log("model declared");
    }
    console.log(ovModels.has(device));
    if (!ovModels.has(device)){
        compiledModel = await core.compileModel(model, device);
        console.log("new device ", ovModels.has(device));
        await ovModels.set(device, compiledModel);
    } else {
        compiledModel = ovModels.get(device);
        console.log("has in map");
    }    
    inferRequest = compiledModel.createInferRequest();
    return inferRequest;
}

let semaphore = false; //semaphore

async function runModel(img, width, height, device){
    // if device in ovModels, use precompiled model, otherwise load and compile model and ut to the map

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 10));
    }

    semaphore = true;

    try{
        // CANVAS TO MAT CONVERSION:
        if (mat == null || mat.data.length != img.data.length){
            mat = new cv.Mat(height, width, cv.CV_8UC4);
        }
        mat.data.set(img.data);

        // MAT PREPROCESSING:
        let preprocessingResult = preprocessMat(mat);
        let preprocessedImage = preprocessingResult.image;
        let paddingInfo = preprocessingResult.paddingInfo;

        // MAT TO OpenVINO TENSOR CONVERSION:
        const tensorData = new Float32Array(preprocessedImage.data);
        const shape = [1, preprocessedImage.rows, preprocessedImage.cols, 3];
        const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

        // MAP -> set (add to map), has (check if map has saved), get (take item from map), key-value type

        // OpenVINO INFERENCE (TO DO)
        const startTime = performance.now();
        inferRequest = await createInferRequest(device);
        // inferRequest.setInputTensor(inputTensor);
        // inferRequest.infer();
        const endTime = performance.now();
        const inferenceTime = endTime - startTime;


        return {
            img : img, 
            inferenceTime : inferenceTime.toFixed(2).toString()
        };
    } finally {
        semaphore = false;
    }
}


function takeTime(){
    return performance.now();
}