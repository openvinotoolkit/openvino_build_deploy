const { addon: ov } = require('openvino-node');

module.exports = { detectDevices }

const core = new ov.Core();

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}