const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

module.exports = { detectDevices, runModel, takeTime, blurImage, addWatermark }

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
const ovModels = new Map(); // compiled models
let model = null; // read model

// variables used to calculate inference time:
let infTimes = [];
let avgInfTime = 0;
let prevDevice = null;

const inputSize = { w: 256, h: 256 };
let outputMask = null;
let ovLogo = null;


async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}


function average(array){
    let sum = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    return (sum / array.length);
}


async function getModelPath() {
    if (fs.existsSync(path.join(__dirname, '../../app.asar'))){     
        //if running compiled program
        return path.join(__dirname, "../../app.asar.unpacked/models/selfie_multiclass_256x256.xml");
    } else {    
        //if running npm start
    return path.join(__dirname, "../models/selfie_multiclass_256x256.xml");
    }
}


function getOvLogo(){
    let imgPath = null;
    if (fs.existsSync(path.join(__dirname, '../../app.asar'))){     
        //if running compiled program
        imgPath = path.join(__dirname, "../../app.asar.unpacked/assets/openvino-logo.png");
    } else {    
        //if running npm start
        imgPath = path.join(__dirname, "../assets/openvino-logo.png");
    }
    return sharp(imgPath)
        .flop()
        .composite([{
            input: Buffer.from([255, 255, 255, 100]), // 50% transparent white
            raw: {
                width: 1,
                height: 1,
                channels: 4
            },
            tile: true,
            blend: 'dest-in'
    }])
}


async function getModel(device) {
    // if model not loaded
    if (model == null) {
        const modelPath = await getModelPath();
        model = await core.readModel(modelPath);
    }

    // if cached
    if (ovModels.has(device)) return ovModels.get(device)

    // compile and cache
    let compiledModel = await core.compileModel(model, device);
    ovModels.set(device, compiledModel);

    return compiledModel;
}


function normalizeArray(array) {
    // Find the minimum and maximum values in the array
    const throughput = 0.5;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < array.length; i++) {
        const val = array[i];

        if (val < min) min = val;
        if (val > max) max = val;
    }

    // If max equals min, all values are the same and can be set to 0 or 1
    if (max === min) return new Array(array.length).fill(0);

    // Normalize each element of the array
    return array.map(value => {
        const coef = (value - min) / (max - min);

        return coef > throughput ? 1 : 0;
    });
}


async function preprocess(originalImg) {
    const inputImg = await originalImg
        .resize(inputSize.w, inputSize.h, { fit: 'fill' })
        .removeAlpha()
        .raw().toBuffer();
    const resizedImageData = new Uint8ClampedArray(inputImg.buffer);
    const tensorData = Float32Array.from(resizedImageData, x => x / 255);
    const shape = [1, inputSize.w, inputSize.h, 3];
    return new ov.Tensor(ov.element.f32, shape, tensorData);
}


function postprocessMask(resultTensor) {
    const channels = 3;
    const normalizedData = normalizeArray(resultTensor.data);
    const imageBuffer = Buffer.alloc(inputSize.w * inputSize.h * channels);

    for (let i = 0; i < normalizedData.length; i += 6) {
        const indexOffset = i / 6 * channels;
        const value = 255 * normalizedData[i];

        imageBuffer[indexOffset] = value;
        imageBuffer[indexOffset + 1] = value;
        imageBuffer[indexOffset + 2] = value;
    }
    return imageBuffer;
}


function calculateAverageInferenceTime(inferenceTime, device) {
    if (prevDevice !== device){
        infTimes = [];
        prevDevice = device;
    }
    if (infTimes.length >= 50){
        infTimes.pop();
    }
    infTimes.unshift(inferenceTime);
    return average(infTimes);
}


async function runModel(img, width, height, device){
    const originalImg = sharp(img.data, { raw: { channels: 4, width, height } });

    let model = await getModel(device);

    const inputTensor = await preprocess(originalImg);

    const startTime = performance.now();            // TIME MEASURING : START

    // OpenVINO INFERENCE
    let inferRequest = model.createInferRequest();

    inferRequest.setInputTensor(inputTensor);
    inferRequest.infer();
    const outputLayer = model.outputs[0];
    const resultTensor = inferRequest.getTensor(outputLayer);

    const endTime = performance.now();              // TIME MEASURING : END
    const inferenceTime = endTime - startTime;

    avgInfTime = calculateAverageInferenceTime(inferenceTime, device)

    // POSTPROCESSING
    outputMask = postprocessMask(resultTensor);

    return {
        width : width,
        height : height,
        inferenceTime : avgInfTime.toFixed(2).toString()
    };
}


async function blurImage(image, width, height) {
    if (outputMask == null)
        return {
            img: image.data,
            width: width,
            height: height
        };

    const person = await sharp(outputMask, {
        raw: {
            channels: 3,
            width: inputSize.w,
            height: inputSize.h,
        }
    })
        .resize(width, height, { fit: 'fill' })
        .unflatten()
        .composite([{
            input: image.data,
            raw: {
                channels: 4,
                width,
                height,
            },
            blend: 'in',
        }])
        .toBuffer();

    const blurSize = Math.floor(width * 0.01)
    const screen = await sharp(image.data, {
        raw: { channels: 4, width, height },
    })
        .blur(blurSize)
        .composite([{
            input: person,
            raw: { channels: 4, width, height },
            blend: 'atop'
        }])
        .raw()
        .toBuffer();

    return {
        img: new Uint8ClampedArray(screen.buffer),
        width: width,
        height: height,
    };
}


async function addWatermark(image, width, height) {
    if (ovLogo == null){
        ovLogo = getOvLogo();
    }
    const watermarkWidth = Math.floor(width * 0.3);
    const watermark = await ovLogo
        .resize({ width: watermarkWidth })
        .toBuffer()

    const result = await sharp(image.data, {
        raw: { channels: 4, width, height },
    })
        .composite([{
            input: watermark,
            gravity: 'southwest'
        }])
        .raw()
        .toBuffer()

    return {
        img: new Uint8ClampedArray(result.buffer),
        width: width,
        height: height,
    };
}


function takeTime(){
    return performance.now();
}
