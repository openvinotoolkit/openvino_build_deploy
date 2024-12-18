const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const os = require('os');

module.exports = { detectDevices, runModel, blurImage, addWatermark, clearWatermarkCache, initializeWatermark }

// Sharp settings
sharp.cache(100);  // Increased cache size
sharp.concurrency(2);  // Allow 2 concurrent operations since we're using Promise.all
sharp.simd(true);  // Ensure SIMD optimizations are enabled

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
if (core.getAvailableDevices().includes('CPU')) {
    core.setProperty("CPU", {
        "INFERENCE_NUM_THREADS": Math.max(2, os.cpus().length - 1),
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_HYPER_THREADING": "YES"
    });
}
const ovModels = new Map(); // compiled models
let model = null; // read model
const watermarkCache = new Map();
let baseWatermark = null;
let isInitialized = false;

const inputSize = { w: 256, h: 256 };
let outputMask = null;

const preprocessBuffer = new Float32Array(inputSize.w * inputSize.h * 3);
const normalizedBuffer = new Float32Array(inputSize.w * inputSize.h * 6);
const inferRequests = new Map();

async function initializeWatermark() {
    
    if (isInitialized) return;
    
    let imgPath = null;
    if (fs.existsSync(path.join(__dirname, '../../app.asar'))) {     
        imgPath = path.join(__dirname, "../../app.asar.unpacked/assets/openvino-logo.png");
    } else {    
        imgPath = path.join(__dirname, "../assets/openvino-logo.png");
    }    
    
    baseWatermark = await sharp(imgPath)
        .flop()
        .composite([{
            input: Buffer.from([255, 255, 255, 100]),
            raw: {
                width: 1,
                height: 1,
                channels: 4
            },
            tile: true,
            blend: 'dest-in'
        }])
        .toBuffer();
    
    isInitialized = true;
}

async function getWatermarkForSize(width) {
    const watermarkWidth = Math.floor(width * 0.3);    
    
    if (watermarkCache.has(watermarkWidth)) {
        return watermarkCache.get(watermarkWidth);
    }    

    if (!isInitialized) {
        await initializeWatermark();
    }
    
    const resizedWatermark = await sharp(baseWatermark)
        .resize({ 
            width: watermarkWidth,
            fastShrinkOnLoad: true,
            kernel: 'cubic' })
        .toBuffer({ resolveWithObject: false });
    
    if (watermarkCache.size > 5) {
        const firstKey = watermarkCache.keys().next().value;
        watermarkCache.delete(firstKey);
    }

    watermarkCache.set(watermarkWidth, resizedWatermark);
    return resizedWatermark;
}

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
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
    const throughput = 0.5;
    let min = Infinity;
    let max = -Infinity;    
    
    for (let i = 0; i < array.length; i++) {
        const val = array[i];
        if (val < min) min = val;
        if (val > max) max = val;
    }
    
    if (max === min) {
        normalizedBuffer.fill(0);
        return normalizedBuffer;
    }
    
    for (let i = 0; i < array.length; i++) {
        const coef = (array[i] - min) / (max - min);
        normalizedBuffer[i] = coef > throughput ? 1 : 0;
    }

    return normalizedBuffer;
}


async function preprocess(originalImg) {
    //console.time('preprocess');
    const inputImg = await originalImg
        .resize(inputSize.w, inputSize.h, { fit: 'fill' })
        .removeAlpha()
        .raw().toBuffer();
    
    // Reuse existing buffer instead of creating new one
    for (let i = 0; i < inputImg.length; i++) {
        preprocessBuffer[i] = inputImg[i] / 255;
    }
    
    const shape = [1, inputSize.w, inputSize.h, 3];
    //console.timeEnd('preprocess');
    return new ov.Tensor(ov.element.f32, shape, preprocessBuffer);
}


function postprocessMask(resultTensor) {
    //console.time('postprocess');
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
    //console.timeEnd('postprocess');
    return imageBuffer;
}

async function getInferRequest(device) {
    if (inferRequests.has(device)) {
        return inferRequests.get(device);
    }
    const model = await getModel(device);
    const inferRequest = model.createInferRequest();
    inferRequests.set(device, inferRequest);
    return inferRequest;
}


async function runModel(img, width, height, device) {
    const originalImg = sharp(img.data, { raw: { channels: 4, width, height } });
    const inputTensor = await preprocess(originalImg);   
    
    const inferRequest = await getInferRequest(device);
    
    const startTime = performance.now() / 1000;
    //console.time('inference');
    inferRequest.setInputTensor(inputTensor);
    inferRequest.infer();
    const outputLayer = (await getModel(device)).outputs[0];
    const resultTensor = inferRequest.getTensor(outputLayer);
    //console.timeEnd('inference');
    const stopTime = performance.now() / 1000;
    const inferenceTime = (stopTime - startTime) * 1000;
        
    outputMask = postprocessMask(resultTensor);
    
    return {
        width: width,
        height: height,
        inferenceTime: inferenceTime
    };
}

async function blurImage(image, width, height) {
    //console.time('blur');
    if (outputMask == null) {
        //console.timeEnd('blur');
        return {
            img: image.data,
            width: width,
            height: height
        };
    }    

    // Increase blur size for more visible effect
    const blurSize = Math.max(10, Math.floor(width * 0.01));
    
    try {        
        const blurPipeline = sharp(image.data, {
            raw: { channels: 4, width, height },
            limitInputPixels: false,
        });

        const maskPipeline = sharp(outputMask, {
            raw: {
                channels: 3,
                width: inputSize.w,
                height: inputSize.h,
            },
            limitInputPixels: false,
        });

        // Run pipelines in parallel
        const [blurred, person] = await Promise.all([
            blurPipeline
                .blur(blurSize)
                .raw()
                .toBuffer({ resolveWithObject: false }),

            maskPipeline
                .resize(width, height, { 
                    fit: 'fill',
                    kernel: 'lanczos3', // Better quality for mask edges
                    fastShrinkOnLoad: true
                })
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
                .raw()
                .toBuffer({ resolveWithObject: false })
        ]);        

        // Final composition
        const screen = await sharp(blurred, {
            raw: { channels: 4, width, height },
            limitInputPixels: false,
        })
            .composite([{
                input: person,
                raw: { channels: 4, width, height },
                blend: 'over'
            }])
            .raw()
            .toBuffer();

        //console.timeEnd('blur');
        return {
            img: new Uint8ClampedArray(screen.buffer),
            width: width,
            height: height,
        };
    } catch (error) {
        console.error('Error in blur processing:', error);
        //console.timeEnd('blur');
        return {
            img: image.data,
            width: width,
            height: height
        };
    }
}

async function addWatermark(image, width, height) {
    try {
        const watermark = await getWatermarkForSize(width);

        const result = await sharp(image.data, {
            raw: { channels: 4, width, height },
        })
            .composite([{
                input: watermark,
                gravity: 'southwest'
            }])
            .raw()
            .toBuffer();

        return {
            img: new Uint8ClampedArray(result.buffer),
            width: width,
            height: height,
        };
    } catch (error) {
        console.error('Error adding watermark:', error);
        // Return original image if watermark fails
        return {
            img: image.data,
            width: width,
            height: height,
        };
    }
}

// Clear cache method for memory management
function clearWatermarkCache() {
    watermarkCache.clear();
    baseWatermark = null;
    isInitialized = false;
    return Promise.resolve();
}