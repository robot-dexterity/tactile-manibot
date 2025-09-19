import argparse
import cv2
import json

from tg3.utils.transform_image import transform_image
from tg3.utils.labeller import RegressionLabeller, LabelledModel
from tg3.utils.contact_model import ContactModel
from tg3.utils.NNmodels import CNN, NatureCNN, ResNet, ResidualBlock
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
    

def setup_model(in_dim, in_channels, out_dim, model_params, device='cpu'):
    """ Import CNN model """

    model_type, kwargs = model_params['model_type'], model_params.get('model_kwargs', {})

    if model_type.startswith(('posenet', 'simple_cnn')):
        model = CNN(in_dim, in_channels, out_dim, device=device, **kwargs).to(device)
        model.apply(model.weights_init_normal)

    elif model_type.startswith('nature_cnn'):
        model = NatureCNN(in_dim, in_channels, out_dim, **kwargs).to(device)
        model.apply(model.weights_init_normal)

    elif model_type.startswith('resnet'):
        model = ResNet(ResidualBlock, in_channels, out_dim, **kwargs).to(device)

    else:
        raise ValueError(f'Incorrect model_type specified: {model_type}')

    return model


def load_model(model_dir):
    """ Set up the pose model and its parameters from the specified directory. """

    model_params = load_json_obj(f"{model_dir}/model_params")
    label_params = load_json_obj(f"{model_dir}/model_label_params")
    image_proc_params = load_json_obj(f"{model_dir}/model_image_params")['image_processing']

    labeller = RegressionLabeller(**label_params)
    in_dim, in_channels, out_dim = image_proc_params['dims'], 1, labeller.out_dim

    model = setup_model(in_dim, in_channels, out_dim, model_params)

    return model, image_proc_params, labeller


def load_json_obj(name):
    with open(name if name.endswith('.json') else name + ".json", "r") as fp:
        return json.load(fp)


def main(model_dir):

    sensor_params = load_json_obj(f"{model_dir}/sensor_image_params.json")
    # sensor_params["source"] = 1 
    sensor = RealSensor(sensor_params)

    model, image_proc_params, labeller = load_model(model_dir)
    pose_model = LabelledModel(model, image_proc_params, labeller)#, ros2_topic=ros2.publish_pose)
    contact_model = ContactModel()#ros2_topic=ros2.publish_contact)

    run_live_loop(sensor, pose_model, contact_model)


if __name__ == "__main__":

    model_dir = f'./tactile_data/ur5_tacpad_1/surface_zRxy_shear/regress_pose_zRxy/simple_cnn'
    # model_dir = f'./tactile_data/mg400_tactip/edge_xRz_shear/regress_pose_xRz/simple_cnn'

    main(model_dir)
