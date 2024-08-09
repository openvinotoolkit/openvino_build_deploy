const { addon: ov } = require('openvino-node');

module.exports = { detectDevices, runModel }

const core = new ov.Core();

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

async function runModel(img){
    return img;
}