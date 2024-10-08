const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const path = require('node:path');
const fs = require('node:fs/promises');
const sharp = require('sharp');

const ModelExecutor = require('./model-executor');

module.exports = { detectDevices, runModel, takeTime, blurImage }

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
let readModel = null;
const ovModels = new Map(); // compiled models
// semaphore used in runModel:
let semaphore = false;
// variables used to calculate inference time:
let infTimes = [];
let avgInfTime = 0;

let isFirst = true; // not counting first iteration to average

const inputSize = { w: 256, h: 256 };
let outputMask = null;

let prevDevice = null;



async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function calculateAverage(array){
    let sum = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    return (sum / array.length);
}

async function getModelPath() {
    const archivePath = path.join(__dirname, '../../app.asar.unpacked/models/selfie_multiclass_f32_256x256.xml');
    const devPath = path.join(__dirname, '../models/selfie_multiclass_f32_256x256.xml');

    try {
        await fs.access(archivePath, fs.constants.F_OK);

        return archivePath;
    } catch(e) {
        return devPath;
    }
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

async function preprocessing(originalImg) {
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
        const indexOffset = i/6 * channels;
        const value = 255*normalizedData[i];

        imageBuffer[indexOffset] = value;
        imageBuffer[indexOffset + 1] = value;
        imageBuffer[indexOffset + 2] = value;
    }
    return imageBuffer;
}

async function runModel(img, width, height, device) {
    const originalImg = sharp(img.data, { raw: { channels: 4, width, height } });

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 7));
    }

    semaphore = true;

    try {
        if (!ovModels.has(device)) {
            if (readModel==null){
                const modelPath = await getModelPath();
                readModel = await core.readModel(modelPath);
            }
            modelExecutor = new ModelExecutor(ov, readModel);
            await modelExecutor.init();
            ovModels.set(device, modelExecutor);
            isFirst = true;
        } else {
            modelExecutor = ovModels.get(device);
        }

        const inputTensor = await preprocessing(originalImg);

        // OpenVINO INFERENCE
        const startTime = performance.now();            // TIME MEASURING : START
        const resultTensor = await modelExecutor.execute(device, [inputTensor]);
        const endTime = performance.now();
        const inferenceTime = endTime - startTime;      // TIME MEASURING : END

        // COUNTING AVERAGE INFERENCE TIME
        if(prevDevice != device){
            infTimes = [];
        }
        if (!isFirst) {
            if (infTimes.length >= 50) infTimes.pop();

            infTimes.unshift(inferenceTime);
            avgInfTime = calculateAverage(infTimes);
        }
        prevDevice = device;

        outputMask = postprocessMask(resultTensor);
        isFirst = false;

        return {
            width,
            height,
            inferenceTime: avgInfTime.toFixed(2).toString()
        };

    } finally {
        semaphore = false;
    }
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

    const screen = await sharp(image.data, {
            raw: { channels: 4, width, height },
        })
        .blur(10)
        .composite([{
            input: person,
            raw: { channels: 4, width, height },
            blend: 'atop'
        }])
        .raw()
        .toBuffer();

    return{
        img: new Uint8ClampedArray(screen.buffer),
        width: width,
        height: height,
    };
}

function takeTime(){
    return performance.now();
}
