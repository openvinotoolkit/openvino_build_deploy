const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');
const path = require('path');
const fs = require('fs');

module.exports = { detectDevices, runModel, takeTime, blurImage }

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
const ovModels = new Map(); // compiled models
let model = null; // read model
let modelBlur = null; // read model for blurring
let compiledBlur = null; // compiled model for blurring
// mats used during preprocessing:
let mat = null;
let resizedMat = null;
let blurredImage = null;
// mats used during postprocessing:
let maskMatOrg = null;
let maskMatSmall = null;
let notMask = null;
let matToBlur = null;
let alpha = null;
let finalMat = null;
// semaphore used in runModel:
let semaphore = false;
// variables used to calculate inference time:
let infTimes = [];
let avgInfTime = 0;


async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}


function preprocessMat(image, targetHeight = 256, targetWidth = 256) {
    // RESIZING
    if (resizedMat == null || resizedMat.size().width !== targetWidth || resizedMat.size().height !== targetHeight){
        resizedMat = new cv.Mat(targetHeight, targetWidth, cv.CV_8UC3);
    }
    cv.resize(image, resizedMat, resizedMat.size());

    //CHANGING FROM 4-CHANNEL BGRA TO 3-CHANNEL RGB
    cv.cvtColor(resizedMat, resizedMat, cv.COLOR_BGRA2RGB);

    return {
        image : resizedMat
    };
}

function convertToMultiDimensionalArray(tensor, shape) {
    function createArray(dim, idx) {
        if (dim >= shape.length) {
            return tensor[idx];
        }
        
        let arr = [];
        let size = shape.slice(dim + 1).reduce((a, b) => a * b, 1);
        
        for (let i = 0; i < shape[dim]; i++) {
            arr.push(createArray(dim + 1, idx + i * size));
        }
        return arr;
    }
    
    return createArray(0, 0);
}


function calculateAverage(array){
    let sum = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    return (sum / array.length);
}


function postprocessMask (mask){
    // LABELING
    const maskShape = mask.getShape();
    const multidimArray = convertToMultiDimensionalArray(mask.data, maskShape);
    const labelMask = multidimArray[0].map(row => row.map(pixel => pixel.indexOf(Math.max(...pixel))));

    // RESIZING
    if (maskMatSmall == null){
        maskMatSmall = new cv.Mat(labelMask.length, labelMask[0].length, cv.CV_8UC1);
    }
    maskMatSmall.data.set(labelMask.flat());
    cv.resize(maskMatSmall, maskMatOrg, maskMatOrg.size(), cv.INTER_NEAREST);
}


async function runModel(img, width, height, device){

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 7));
    }

    semaphore = true;
    let isFirst = false; // not counting first iteration to average

    try{
        // CANVAS TO MAT CONVERSION:
        if (mat == null || mat.data.length !== img.data.length){
            mat = new cv.Mat(height, width, cv.CV_8UC4);
        }
        mat.data.set(img.data);

        // MAT PREPROCESSING:
        let preprocessingResult = preprocessMat(mat);
        let preprocessedImage = preprocessingResult.image;

        // MAT TO OpenVINO TENSOR CONVERSION:
        const tensorData = Float32Array.from(preprocessedImage.data, x => x / 255.0);
        const shape = [1, preprocessedImage.rows, preprocessedImage.cols, 3];
        const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

        // OpenVINO INFERENCE
        const startTime = performance.now();            // TIME MEASURING : START

        let compiledModel, inferRequest;
        if (model == null){
            if (fs.existsSync(path.join(__dirname, '../../app.asar'))){     //if running compiled program
                model = await core.readModel(path.join(__dirname, "../../app.asar.unpacked/models/selfie_multiclass_256x256.xml"));
            } else {    //if running npm start
            model = await core.readModel(path.join(__dirname, "../models/selfie_multiclass_256x256.xml"));
            }
        }
        if (!ovModels.has(device)){
            compiledModel = await core.compileModel(model, device);
            ovModels.set(device, compiledModel);
            isFirst = true;
        } else {
            compiledModel = ovModels.get(device);
        }
        inferRequest = compiledModel.createInferRequest();
        inferRequest.setInputTensor(inputTensor);
        inferRequest.infer();
        const outputLayer = compiledModel.outputs[0];
        const resultInfer = inferRequest.getTensor(outputLayer);

        const endTime = performance.now();
        const inferenceTime = endTime - startTime;      // TIME MEASURING : END

        // COUNTING AVERAGE INFERENCE TIME
        if(!isFirst){
            if(infTimes.length>=50){
                infTimes.pop();
            }
            infTimes.unshift(inferenceTime);
            avgInfTime = calculateAverage(infTimes);
        }

        // POSTPROCESSING
        if (maskMatOrg == null || mat.rows !== maskMatOrg.rows || mat.cols !== maskMatOrg.cols){
            maskMatOrg = new cv.Mat(height, width, cv.CV_8UC1);
        }
        postprocessMask(resultInfer);

        // MASK PREPARATION
        cv.threshold(maskMatOrg, maskMatOrg, 0, 255, cv.THRESH_BINARY);
        if (notMask == null || notMask.data.length !== maskMatOrg.data.length){
            notMask = new cv.Mat(height, width, cv.CV_8UC1);
        }
        cv.bitwise_not(maskMatOrg, notMask);
        cv.threshold(notMask, notMask, 254, 255, cv.THRESH_BINARY);

        return {
            width : maskMatOrg.cols,
            height : maskMatOrg.rows,
            inferenceTime : avgInfTime.toFixed(2).toString()
        };

    } finally {
        semaphore = false;
    }
}


async function blurImage(image, width, height) {
    if (maskMatOrg == null){
        return{
            img : image.data,
            width : width,
            height : height
        };
    }
    // MAT FROM IMAGE DATA (from webcam)
    // if (matToBlur == null || matToBlur.data.length !== image.data.length){
    //     matToBlur = new cv.Mat(height, width, cv.CV_8UC4);
    // }
    // matToBlur.data.set(image.data);

    // CONVERSION TO OpenVINO TENSOR:
    const tensorDataMask = Float32Array.from(maskMatOrg.data, x => x / 255.0);
    const inputTensorMask = new ov.Tensor(ov.element.f32, [1, maskMatOrg.rows, maskMatOrg.cols, 1], tensorDataMask);

    const tensorDataImg = Float32Array.from(image.data);
    const inputTensorImg = new ov.Tensor(ov.element.f32, [1, height, width, 4], tensorDataImg);

    // OpenVINO BLURRING MODEL DECLARATION:
    if (modelBlur == null){
        if (fs.existsSync(path.join(__dirname, '../../app.asar'))){     //if running compiled program
            modelBlur = await core.readModel(path.join(__dirname, "../../app.asar.unpacked/models/postproc_model.xml"));
        } else {    //if running npm start
        modelBlur = await core.readModel(path.join(__dirname, "../models/postproc_model.xml"));
        }
    }
    if (compiledBlur == null){
        compiledBlur = await core.compileModel(modelBlur, "AUTO");
    }

    let begin = performance.now()
    // OpenVINO BLURRING MODEL INFERENCE:
    const inferRequest = compiledBlur.createInferRequest();
    inferRequest.setInputTensor(0, inputTensorImg);
    inferRequest.setInputTensor(1, inputTensorMask);
    inferRequest.infer();
    const outputLayer = compiledBlur.outputs[0];
    const resultInfer = inferRequest.getTensor(outputLayer);
    console.log("time: ", performance.now() - begin, " ms");

    // // MERGING IMAGES
    // if (finalMat == null || image.data.length !== finalMat.data.length){
    //     finalMat = new cv.Mat(height, width, cv.CV_8UC4);
    // }
    // finalMat.data.set(resultData);

    return{
        img : new Uint8ClampedArray(resultInfer.data),
        width : width,
        height : height
    };
}


function takeTime(){
    return performance.now();
}