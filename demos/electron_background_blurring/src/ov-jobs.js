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
let blurredImage = null;
let maskMatSmall = null
let maskMatOrg = null

let model = null;

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function preprocessMat(image, targetHeight = 256, targetWidth = 256) {
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

function getTensorIndex(n, h, w, c) {
    return ((n * shape[1] + h) * shape[2] + w) * shape[3] + c;
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


function postprocessMask (mask, padInfo){
    // TAKE OUT LABELS
    const maskShape = mask.getShape();
    const multidimArray = convertToMultiDimensionalArray(mask, maskShape);
    const labelMask = multidimArray[0].map(row => row.map(pixel => pixel.indexOf(Math.max(...pixel))));    

    // UNPADDING
    const unpadH = maskShape[1] - padInfo.bottomPadding;
    const unpadW = maskShape[2] - padInfo.rightPadding;
    const labelMaskUnpadded = labelMask.slice(0, unpadH).map(row => row.slice(0, unpadW));


    // RESIZING
    const labelMaskResized = new Array(maskMatOrg.size()[0]).fill(0).map(() => new Array(maskMatOrg.size()[1]).fill(0));
    maskMatOrg.data.set(labelMaskResized);
    /*
    mat resize:  5.1266000010073185
    array resize:  0.07750000059604645
    */
}

let semaphore = false; 

async function runModel(img, width, height, device){

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 7));
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

        // MAT TO OpenVINO TENSOR CONVERSION:
        const tensorData = new Float32Array(preprocessedImage.data.length);
        for (let i = 0; i < preprocessedImage.data.length; i++) {
            tensorData[i] = preprocessedImage.data[i] / 255.0;
        }
        const shape = [1, preprocessedImage.rows, preprocessedImage.cols, 3];
        const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

        // OpenVINO INFERENCE
        const startTime = performance.now();            // TIME MEASURING : START
        let compiledModel = null;
        if (model == null){
            model = await core.readModel(path.join(__dirname, "../models/selfie_multiclass_256x256.xml"));
        }
        if (!ovModels.has(device)){
            compiledModel = await core.compileModel(model, device);
            ovModels.set(device, compiledModel);
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

        // BLURRING IMAGE (TO DO)
        if (maskMatOrg == null){
            maskMatOrg = new cv.Mat(height, width, cv.CV_8UC1);
        }
        postprocessMask(resultInfer, preprocessingResult.paddingInfo);
        // if (blurredImage == null){
        //     blurredImage = new cv.Mat(height, width, cv.CV_8UC3);
        // }
        // cv.GaussianBlur(mat, blurredImage, new cv.Size(55,55), 0);


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