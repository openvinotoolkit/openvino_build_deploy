
const { performance } = require('perf_hooks');

class ModelExecutor {
    initialized = false;
    core = null;
    model = null;
    compiledModel = null;
    ir1 = null;
    ir2 = null;
    lastUsedDevice = null;

    nextInferRequestNum = 1;

    constructor(ov, model) {
        this.core = new ov.Core();
        this.model = model;
    }

    async init() {
        this.initialized = true;
    }

    async compileModel(device = 'AUTO') {
        if (device != "GPU"){
            this.compiledModel = await this.core.compileModel(
                this.model, device,
                {
                  'PERFORMANCE_HINT': 'LATENCY',
                  'NUM_STREAMS': 2
                },
            );
        } else {
            this.compiledModel = await this.core.compileModel(
                this.model, device,
                {
                  'PERFORMANCE_HINT': 'LATENCY'
                },
        );
        }
        this.lastUsedDevice = device;

        return this.compiledModel;
    }

    async execute(device, inputData) {
      const begin = performance.now();
      let useTwoInferRequests;
        if (device == "GPU"){
            useTwoInferRequests = false;
        } else {
            useTwoInferRequests = true;
        }

        if (!this.initialized)
            throw new Error('Model isn\'t initialized');

        if (!this.compiledModel || device !== this.lastUsedDevice) {
            await this.compileModel(device);
            this.ir1 = await this.compiledModel.createInferRequest();

            if (useTwoInferRequests)
              this.ir2 = await this.compiledModel.createInferRequest();
        }

        const result = this.getInferRequest(useTwoInferRequests).infer(inputData);
        const keys = Object.keys(result);
        // console.log(performance.now()-begin);
        return result[keys[0]];
    }

    getInferRequest(useTwo = false) {
      if (!useTwo) return this.ir1;

      switch(this.nextInferRequestNum) {
        case 1:
          this.nextInferRequestNum = 2;
          return this.ir1;
        case 2:
          this.nextInferRequestNum = 1;
          return this.ir2;
      }
    }
}

module.exports = ModelExecutor;
