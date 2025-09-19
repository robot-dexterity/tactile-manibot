"""
Use (shift, ctrl, arrow keys) to move robot
"""
import argparse
import cv2
import json

from tg3.data.utils.transform_image import transform_image
from tg3.learning.image_to_feature.cnn.labeller import RegressionLabeller, LabelledModel
from tg3.learning.image_to_feature.train import setup_model
from tg3.tasks.utils.contact_model import ContactModel
# import tg3.tasks.utils.ros2_handler as ros2


def run_live_loop(sensor, pose_model, contact_model, num_iterations=10000):

    contact_model.reference = sensor.process()

    for i in range(num_iterations):

        tactile_image = sensor.process()
        pred_pose = pose_model.predict(tactile_image) 
        contact, ssim = contact_model.predict(tactile_image)

        cv2.imshow('Tactile Image', tactile_image); cv2.waitKey(1)
        print(f"{i+1}/{num_iterations} contact={contact} | SSIM={ssim:.2f} | ", end='')
        print(f" pose={[f'{p:.1f}' for p in pred_pose[:6]]} ".replace("'",""))

    # ros2.shutdown()


class RealSensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params
        source = sensor_params.get('source', 0)
        exposure = sensor_params.get('exposure', -7)

        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):
            self.cam.read()  # Hack - initial camera transient

    def read(self):
        _, img = self.cam.read()
        return img

    def process(self, outfile=None):
        img = transform_image(self.read(), **self.sensor_params)
        if outfile:
            cv2.imwrite(outfile, img)
        return img
    

def load_model(model_dir):
    """ Set up the pose model and its parameters from the specified directory. """

    model_params = load_json_obj(f"{model_dir}/model_params")
    label_params = load_json_obj(f"{model_dir}/model_label_params")
    image_proc_params = load_json_obj(f"{model_dir}/model_image_params")['image_processing']

    labeller = RegressionLabeller(**label_params)
    in_dim, in_channels, out_dim = image_proc_params['dims'], 1, labeller.out_dim

    model = setup_model(in_dim, in_channels, out_dim, model_params, model_dir)

    return model, image_proc_params, labeller


def load_json_obj(name):
    with open(name if name.endswith('.json') else name + ".json", "r") as fp:
        return json.load(fp)


def main(model_dir):

    sensor_params = load_json_obj(f"{model_dir}/sensor_image_params.json")
    sensor_params["source"] = 0 
    sensor = RealSensor(sensor_params)

    model, image_proc_params, labeller = load_model(model_dir)
    pose_model = LabelledModel(model, image_proc_params, labeller)#, ros2_topic=ros2.publish_pose)
    contact_model = ContactModel()#ros2_topic=ros2.publish_contact)

    run_live_loop(sensor, pose_model, contact_model)


def parse(
    path = './tactile_data',
    robot = 'mg400',
    sensor = 'tactip',
    experiment='edge_xRz_shear',
    predict='pose_xRz',
    model='simple_cnn',
):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=path, help='Root directory for tactile data')
    parser.add_argument('-r', '--robot', type=str, default=robot, help='Robot type (e.g., sim)')
    parser.add_argument('-s', '--sensor', type=str, default=sensor, help='Sensor type (e.g., tactip)')
    parser.add_argument('-e', '--experiment', type=str, default=experiment, help='Experiment name')
    parser.add_argument('-t', '--predict', type=str, default=predict, help='Prediction target (e.g., pose_yRz)')
    parser.add_argument('-m', '--model', type=str, default=model, help='Model name (e.g., simple_cnn)')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    base_dir = f"{args.path}/{args.robot}_{args.sensor}/{args.experiment}"
    model_dir = f"{base_dir}/regress_{args.predict}/{args.model}"

    main(model_dir)
