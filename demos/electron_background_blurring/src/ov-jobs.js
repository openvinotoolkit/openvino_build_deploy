const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');

module.exports = { detectDevices, runModel }

const core = new ov.Core();
const ovModels = new Map()

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

async function runModel(img, device){
    // if device in ovModels, use precompiled model, otherwise load and compile model and ut to the map

    return img;
}