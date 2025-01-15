from analog.paddle import AnalogPaddle
from analog.yolo import AnalogYolo
import argparse
import cv2
import os
import json


def main(img_path: str, config_file: str):
    output_dir = os.path.abspath(os.path.dirname(img_path))
    with open(config_file) as f:
        config = json.load(f)
    if len(config["model_config"]["detector"]["model_shape"]) == 1:
        meter_reader = AnalogYolo(config, output_dir)
    else:
        meter_reader = AnalogPaddle(config, output_dir)
    image = cv2.imread(img_path)
    det_resutls = meter_reader.detect(image)
    seg_resutls = meter_reader.segment(det_resutls)
    post_resutls = meter_reader.postprocess(seg_resutls)
    meter_reader.reading(post_resutls, image)
    print(f"result images saved to \"{output_dir}\".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-i', '--input', default="data/test.jpg", type=str,
                      help='Required. Path to an image file.')
    parser.add_argument('-c', '--config',  default="config/yolov8.json", type=str,
                      help='Required. config file path')
    parser.add_argument('-t', '--task', default='analog', type=str,
                      help='Required. mode of meter reader, digital or analog')
    args = parser.parse_args()

    main(args.input, args.config)
