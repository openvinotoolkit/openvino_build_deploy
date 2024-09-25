const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const path = require('path');
const fs = require('node:fs/promises');
const sharp = require('sharp');

module.exports = { detectDevices, runModel, takeTime, blurImage }

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
// semaphore used in runModel:
let semaphore = false;
// variables used to calculate inference time:
let infTimes = [];
let avgInfTime = 0;

let outputMask = null;


async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function calculateAverage(array){
    let sum = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    return (sum / array.length);
}

class ModelExecutor {
    initialized = false;
    core = null;
    model = null;
    compiledModel = null;
    ir = null;
    modelFilePath = null;
    lastUsedDevice = null;

    constructor(ov, modelFilePath) {
        this.core = new ov.Core();
        this.modelFilePath = modelFilePath;
    }

    async init() {
        this.model = await core.readModel(this.modelFilePath);
        this.initialized = true;
    }

    async compileModel(device = 'AUTO') {
        this.compiledModel = await this.core.compileModel(this.model, device);
        this.lastUsedDevice = device;

        return this.compiledModel;
    }

    async execute(device, inputData) {
        if (!this.initialized)
            throw new Error('Model isn\'t initialized');

        if (!this.compiledModel || device !== this.lastUsedDevice) {
            await this.compileModel(device);
            this.ir = await this.compiledModel.createInferRequest();
        }

        const result = await this.ir.inferAsync(inputData);
        const keys = Object.keys(result);

        return result[keys[0]];
    }
}

let modelExecutor = null;

async function getModelPath() {
    const archivePath = path.join(__dirname, '../../app.asar.unpacked/models/selfie_multiclass_256x256.xml');
    const devPath = path.join(__dirname, '../models/selfie_multiclass_256x256.xml');

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

async function runModel(img, width, height, device) {
    const inputSize = { w: 256, h: 256 };
    const originalImg = sharp(img.data, { raw: { channels: 4, width, height } });
    const inputImg = await originalImg
        .resize(inputSize.w, inputSize.h, { fit: 'fill' })
        .removeAlpha()
        .raw().toBuffer();
    const resizedImageData = new Uint8ClampedArray(inputImg.buffer);

    while (semaphore) {
        await new Promise(resolve => setTimeout(resolve, 7));
    }

    semaphore = true;
    let isFirst = false; // not counting first iteration to average

    try {
        if (!modelExecutor) {
            const modelPath = await getModelPath();

            modelExecutor = new ModelExecutor(ov, modelPath);
            await modelExecutor.init();
        }
        const tensorData = Float32Array.from(resizedImageData, x => x / 255);
        const shape = [1, inputSize.w, inputSize.h, 3];
        const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

        // OpenVINO INFERENCE
        const startTime = performance.now();            // TIME MEASURING : START
        const resultTensor = await modelExecutor.execute(device, [inputTensor]);
        const endTime = performance.now();
        const inferenceTime = endTime - startTime;      // TIME MEASURING : END

        // COUNTING AVERAGE INFERENCE TIME
        if (!isFirst) {
            if (infTimes.length >= 50) infTimes.pop();

            infTimes.push(inferenceTime);
            avgInfTime = calculateAverage(infTimes);
        }

        const channels = 4;
        const normalizedData = normalizeArray(resultTensor.data);
        const imageBuffer = Buffer.alloc(inputSize.w * inputSize.h * channels);

        for (let i = 0; i < normalizedData.length; i += 6) {
            const indexOffset = i/6 * channels;

            imageBuffer[indexOffset] = 0;
            imageBuffer[indexOffset + 1] = 0;
            imageBuffer[indexOffset + 2] = 0;
            imageBuffer[indexOffset + 3] = 255*normalizedData[i];
        }

        outputMask = imageBuffer;

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

    const mask = await sharp(outputMask, {
            raw: { channels: 4, width: 256, height: 256 }
        })
        .resize(width, height, { fit: 'fill' }).toBuffer();
    const combined = await sharp(image.data, {
            raw: {
                channels: 4,
                width,
                height,
            }
        })
        // .flatten({ background: mask })
        .composite([{ input: mask, raw: { channels: 4, width, height } }])
        .raw().toBuffer();

    return{
        img: new Uint8ClampedArray(combined.buffer),
        width: width,
        height: height,
    };
}

function takeTime(){
    return performance.now();
}
