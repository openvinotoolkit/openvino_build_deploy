const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');

module.exports = { detectDevices, runModel }

const core = new ov.Core();
const ovModels = new Map()

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

function runModel(img, device){
    // if device in ovModels, use precompiled model, otherwise load and compile model and ut to the map
    const startTime = performance.now();
    img = img;
    var i = 3000;
    while(i>0){
        i--;
    }
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;
    return {
        img : img, 
        inferenceTime : inferenceTime.toFixed(2).toString()
    };
}