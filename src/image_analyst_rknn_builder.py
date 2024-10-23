from rknn.api import RKNN
import requests
import os

WEIGHTS_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-coco.weights"
CONFIG_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-coco.cfg"
TINY_WEIGHTS_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-tiny-coco.weights"
TINY_CONFIG_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-tiny-coco.cfg"

DARKNET_NAME = "cv2-yolov3-coco"
RKNN_NAME = "rknn-yolov3-coco"
TINY_DARKNET_NAME = "cv2-yolov3-tiny-coco"
TINY_RKNN_NAME = "rknn-yolov3-tiny-coco"

def download(url: str, filename: str) -> str:
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, "wb") as file:
        file.write(requests.get(url).content)
    return filepath


def build(weights_url: str, config_url: str, darknet_name: str, rknn_name: str):
    weights_path = download(weights_url, f"{darknet_name}.weights")
    config_path = download(config_url, f"{darknet_name}.cfg")

    print(f"Converting model {darknet_name} to {rknn_name}")

    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[0, 0, 0],
        std_values=[255, 255, 255],
        target_platform="rk3588",
    )

    return_code = rknn.load_darknet(model=config_path, weight=weights_path)

    if return_code != 0:
        print("Load darknet model failed!")
        exit(return_code)

    print("Load darknet model done")

    return_code = rknn.build(do_quantization=False)

    if return_code != 0:
        print("Build model failed!")
        exit(return_code)

    print("Build model done")

    return_code = rknn.export_rknn(f"{rknn_name}.rknn")

    if return_code != 0:
        print("Export rknn model failed!")
        exit(return_code)

    print("Export rknn model done")


def main():
    build(WEIGHTS_URL, CONFIG_URL, DARKNET_NAME, RKNN_NAME)
    build(TINY_WEIGHTS_URL, TINY_CONFIG_URL, TINY_DARKNET_NAME, TINY_RKNN_NAME)


if __name__ == "__main__":
    main()
