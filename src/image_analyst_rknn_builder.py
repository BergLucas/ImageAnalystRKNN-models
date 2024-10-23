from rknn.api import RKNN
import requests
import os

WEIGHTS_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-coco.weights"
CONFIG_URL = "https://github.com/BergLucas/ImageAnalystCV2-models/releases/download/v1.0.0/cv2-yolov3-coco.cfg"

DARKNET_NAME = "cv2-yolov3-coco"
RKNN_NAME = "rknn-yolov3-coco"

def download(url: str, filename: str) -> str:
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, "wb") as file:
        file.write(requests.get(url).content)
    return filepath


def main():
    weights_path = download(WEIGHTS_URL, f"{DARKNET_NAME}.weights")
    config_path = download(CONFIG_URL, f"{DARKNET_NAME}.cfg")

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

    return_code = rknn.export_rknn(f"{RKNN_NAME}.rknn")

    if return_code != 0:
        print("Export rknn model failed!")
        exit(return_code)

    print("Export rknn model done")


if __name__ == "__main__":
    main()
