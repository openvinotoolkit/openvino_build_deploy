const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');
const { performance } = require('perf_hooks');

module.exports = { detectDevices, runModel }

const core = new ov.Core();

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

async function runModel(img, device){
    return img;
}