const fs = require('fs').promises;
const path = require('path');
const { addon: ov } = require('openvino-node');
const { cv } = require('opencv-wasm');

module.exports = { detectDevices }

const core = new ov.Core();

async function detectDevices() {
    const devices = core.getAvailableDevices();
    return devices;
}