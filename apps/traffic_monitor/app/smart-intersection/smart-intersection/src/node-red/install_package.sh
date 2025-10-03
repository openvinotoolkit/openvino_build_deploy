#!/bin/bash

if ! npm list -g node-red-contrib-influxdb | grep -q "node-red-contrib-influxdb"; then
    npm install node-red-contrib-influxdb
    apk update
fi

exit 0

