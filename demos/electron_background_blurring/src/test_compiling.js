const { addon: ov } = require('openvino-node');
const path = require('path');


async function run(){   
    let core = new ov.Core();
    let model = await core.readModel(path.join(__dirname, "../models/postproc_model.xml"));
    console.log(model);
    let compiledModel = await core.compileModel(model, "AUTO");
    console.log("model compiled");
}

run();