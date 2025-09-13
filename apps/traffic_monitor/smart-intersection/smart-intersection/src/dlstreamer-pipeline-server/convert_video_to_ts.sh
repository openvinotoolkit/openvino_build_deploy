#!/bin/sh

# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

# script to convert mp4 files in sample-data directory
# to ts files so that gstreamer pipeline can keep running the files
# in infinite loop without having to deallocate buffers

docker pull intel/intel-optimized-ffmpeg:latest

DIRNAME=${PWD}
SAMPLE_DATA_DIRECTORY=${DIRNAME}/sample_data
FFMPEG_DIR="/app/data"
FFMPEG_IMAGE="intel/intel-optimized-ffmpeg:latest"
EXTENSION=${1:-mp4}
PATTERN="*.${EXTENSION}"

DOCKER_RUN_CMD_PREFIX="docker run --rm -v ${SAMPLE_DATA_DIRECTORY}:${FFMPEG_DIR} \
            --entrypoint /bin/sh ${FFMPEG_IMAGE}"

for mfile in "$SAMPLE_DATA_DIRECTORY"/$PATTERN; do
    basefile=$(basename -s .$EXTENSION $mfile)
    tsfile=${SAMPLE_DATA_DIRECTORY}/${basefile}.ts
    echo $tsfile
    if [ -f $tsfile ]; then
        echo "skipping $basefile as $tsfile is available already"
    else
        ffmpegcmd="/opt/build/bin/ffmpeg -i ${FFMPEG_DIR}/${basefile}.${EXTENSION} -c copy ${FFMPEG_DIR}/${basefile}.ts"
        cmd="$DOCKER_RUN_CMD_PREFIX -c '$ffmpegcmd'"
        eval $cmd
    fi
done

