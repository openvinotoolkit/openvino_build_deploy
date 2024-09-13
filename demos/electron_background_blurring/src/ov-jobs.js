const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');
const path = require('path');
const fs = require('fs');
const StackBlur = require( 'stackblur-canvas' );

module.exports = { detectDevices, runModel, takeTime, blurImage }

const core = new ov.Core();
const ovModels = new Map();
let mat = null;
let resizedMat = null;
let paddedImg = null;
let blurredImage = null;
let maskMatOrg = null;
let maskMatSmall = null;
let notMask = null;
let finalMat = null;
let alpha = null;

let infTimes = [];
let avgInfTime = 0;

let model = null;

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
    // TAKE OUT LABELS
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

let semaphore = false; 

let countRun = 0;
let countBlur = 0;

async function runModel(img, width, height, device){

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 7));
    }

    semaphore = true;
    let isFirst = false;

    try{
        const begin = performance.now();

        // CANVAS TO MAT CONVERSION:
        if (mat == null || mat.data.length !== img.data.length){
            mat = new cv.Mat(height, width, cv.CV_8UC4);
        }
        mat.data.set(img.data);
        console.log(performance.now()-begin, "canvas to mat converted");

        // MAT PREPROCESSING:
        let preprocessingResult = preprocessMat(mat);
        let preprocessedImage = preprocessingResult.image;
        console.log(performance.now()-begin, "mat preprocessed");

        // MAT TO OpenVINO TENSOR CONVERSION:
        const tensorData = Float32Array.from(preprocessedImage.data, x => x / 255.0);
        const shape = [1, preprocessedImage.rows, preprocessedImage.cols, 3];
        const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);
        console.log(performance.now()-begin, "mat to tensor");

        // OpenVINO INFERENCE
        const startTime = performance.now();            // TIME MEASURING : START
        let compiledModel, inferRequest;
        if (model == null){
            if (fs.existsSync(path.join(__dirname, '../../app.asar'))){
                model = await core.readModel(path.join(__dirname, "../../app.asar.unpacked/models/selfie_multiclass_256x256.xml"));
            } else {
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
        console.log(performance.now()-begin, "inference");

        // COUNTING AVERAGE INFERENCE TIME
        if(!isFirst){
            if(infTimes.length>=50){
                infTimes.pop();
            }
            infTimes.unshift(inferenceTime);
            avgInfTime = calculateAverage(infTimes);
            console.log("average: ", avgInfTime);
            console.log(performance.now()-begin, "calculating average");
        }

        // POSTPROCESSING
        if (maskMatOrg == null || mat.rows !== maskMatOrg.rows || mat.cols !== maskMatOrg.cols){
            maskMatOrg = new cv.Mat(height, width, cv.CV_8UC1);
        }
        postprocessMask(resultInfer);
        console.log(performance.now()-begin, "postprocessing");

        // MASK PREPARATION
        cv.threshold(maskMatOrg, maskMatOrg, 0, 255, cv.THRESH_BINARY);
        console.log(performance.now()-begin, "threshold");

        if (notMask == null || notMask.data.length !== maskMatOrg.data.length){
            notMask = new cv.Mat(height, width, cv.CV_8UC1);
        }
        cv.bitwise_not(maskMatOrg, notMask);
        cv.threshold(notMask, notMask, 254, 255, cv.THRESH_BINARY);
        console.log(performance.now()-begin, "not mask declared");

        countRun++;
        console.log("\nmodel run:", countRun,"time:", performance.now()-begin, "\n");

        return {
            width : maskMatOrg.cols,
            height : maskMatOrg.rows,
            inferenceTime : avgInfTime.toFixed(2).toString()
        };

    } finally {
        semaphore = false;
    }
}

let matToBlur = null;
let smallImage = null;

async function blurImage(image, width, height){
    const begin = performance.now();
    
    if (matToBlur == null || matToBlur.data.length !== image.data.length){
        matToBlur = new cv.Mat(height, width, cv.CV_8UC4);
    }
    matToBlur.data.set(image.data);

    if (blurredImage == null || matToBlur.data.length !== blurredImage.data.length){
        blurredImage = new cv.Mat(height, width, cv.CV_8UC4);
    }
    
    blurredImage.data.set(image.data);
    cv.blur(matToBlur, blurredImage, new cv.Size(25,25));

    if (finalMat == null || matToBlur.data.length !== finalMat.data.length){
        finalMat = new cv.Mat(height, width, cv.CV_8UC4);
    }
    // console.log(performance.now()-begin, "final mat declared");

    if (alpha == null || matToBlur.data.length !== alpha.data.length) {
        alpha = new cv.Mat(height, width, matToBlur.type(), new cv.Scalar(0, 0, 0, 0)); 
    }

    cv.bitwise_and(matToBlur, alpha, matToBlur, mask=notMask);
    // console.log(performance.now()-begin, "AND org");
    cv.bitwise_and(blurredImage, alpha, blurredImage, mask=maskMatOrg);
    // console.log(performance.now()-begin, "AND blur");

    cv.add(matToBlur, blurredImage, finalMat);
    // console.log(performance.now()-begin, "ADD final");

    countBlur++;
    console.log("\nmodel blur:", countBlur,"time:", performance.now()-begin, "\n");

    return{
        img : new Uint8ClampedArray(finalMat.data),
        width : finalMat.cols,
        height : finalMat.rows
    };
}


function takeTime(){
    return performance.now();
}