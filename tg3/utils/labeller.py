from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from tg3.utils.transform_image import transform_image


class Labeller(ABC):
    """ Abstract base class for label encoders. """

    def __repr__(self):      
        return "{} ({})".format(self.__class__.__name__, self.info)

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def type(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def out_dim(self):
        pass

    @abstractmethod
    def encode(self, labels_dict):
        """ Process label data to NN friendly label for prediction.
        Returns: torch tensor that will be predicted by the NN. """
        pass

    @abstractmethod
    def decode(self, outputs):
        """ Process NN predictions to labels, always decodes to cpu.
        Returns: Dict of np arrays in suitable format for downstream task. """
        pass

    @abstractmethod
    def print_metrics(self, metrics):
        """ Formatted print of metrics given by calc_metrics. """
        pass

    @abstractmethod
    def write_metrics(self, writer, metrics, epoch, mode='val'):
        """ Write metrics given by calc_metrics to tensorboard. """
        pass

    @abstractmethod
    def calc_metrics(self, labels, predictions):
        """ Calculate metrics useful for measuring progress throughout training. """
    pass

    @abstractmethod
    def acc_metric(self, labels, predictions):
        pass


class ClassificationLabeller(Labeller):

    def __init__(self, label_names, device='cpu'):
        self.device = device
        self.label_names = self.target_names = label_names
        self.tolerences = np.ones_like(label_names, dtype=float)

    @property
    def out_dim(self):
        return len(self.target_names)

    def encode(self, labels_dict):
        return torch.eye(self.out_dim, device=self.device)[labels_dict['id']]

    def decode(self, outputs):
        ids = outputs.argmax(1).cpu().numpy()
        return {'id': ids, 'label': np.array(self.target_names)[ids]}

    def print_metrics(self, metrics):
        print('Accuracy:', dict(zip(self.target_names, np.diag(metrics['conf_mat']))))

    def write_metrics(self, writer, metrics, epoch, mode='val'):
        pass

    def calc_metrics(self, labels, predictions):
        return {'conf_mat': self.acc_metric(labels, predictions)}

    def acc_metric(self, labels, predictions):
        """ Accuracy metric for classification problem, returns normalized confusion matrix. """
        conf_mat = confusion_matrix(predictions['id'], labels['id'])
        return conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-8)


class RegressionLabeller(Labeller):

    def __init__(self, label_names, target_label_names, llims, ulims, 
                periodic_label_names=None, target_weights=None, tolerances=None, device='cpu'):
        self.device = device
        self.label_names = label_names
        self.target_names = [name for name in target_label_names if name]

        self.periodic_names = periodic_label_names or []
        self.target_weights = target_weights or np.ones_like(self.target_names, dtype=float)
        self.tolerances = tolerances or np.ones_like(self.target_names, dtype=float)

        self.llims_np, self.ulims_np = np.array(llims), np.array(ulims)
        self.llims_torch = torch.tensor(self.llims_np, dtype=torch.float32, device=self.device)
        self.ulims_torch = torch.tensor(self.ulims_np, dtype=torch.float32, device=self.device)

    @property
    def out_dim(self):
        periodic_dims = sum(self.target_names.count(p) for p in self.periodic_names)
        return len(self.target_names) + periodic_dims

    def encode_norm(self, target, label_name):
        idx = self.label_names.index(label_name)
        llim, ulim = self.llims_torch[idx], self.ulims_torch[idx]
        return ((target - llim) / (ulim - llim) * 2 - 1).unsqueeze(1)

    def decode_norm(self, prediction, label_name):
        idx = self.label_names.index(label_name)
        llim, ulim = self.llims_np[idx], self.ulims_np[idx]
        return ((prediction + 1) * 0.5 * (ulim - llim)) + llim

    def encode_circnorm(self, target):
        ang = torch.deg2rad(target).to(self.device)
        return [torch.sin(ang).unsqueeze(1), torch.cos(ang).unsqueeze(1)]

    def decode_circnorm(self, vec_prediction):
        return torch.atan2(*vec_prediction) * 180 / np.pi

    def encode(self, labels_dict):
        """ Process raw pose data to NN friendly label for prediction.
        Default: maps to weight * range(-1,+1)
        Periodic: maps to weight * [cos angle, sin angle] """
        encoded_pose = []
        for label_name, weight in zip(self.target_names, self.target_weights):
            target = labels_dict[label_name].float().to(self.device)
            if label_name in self.periodic_names:
                encoded_pose += [weight * p for p in self.encode_circnorm(target)]
            else:
                encoded_pose.append(weight * self.encode_norm(target, label_name))
        return torch.cat(encoded_pose, 1)

    def decode(self, outputs):
        decoded_pose, ind = {}, 0
        for label_name, weight in zip(self.target_names, self.target_weights):
            if label_name not in self.periodic_names:
                decoded_pose[label_name] = self.decode_norm(outputs[:, ind].detach().cpu() / weight, label_name) #! breaks if outputs is not 2D, batch size=1
                ind += 1
            else:
                vec = [outputs[:, ind].detach().cpu() / weight, outputs[:, ind+1].detach().cpu() / weight]
                decoded_pose[label_name] = self.decode_circnorm(vec)
                ind += 2
        return decoded_pose

    def print_metrics(self, metrics):
        err_df, acc_df = metrics['err'], metrics['acc']
        print(f"error: {err_df[self.target_names].mean().round(2).to_dict()}")
        print(f"accuracy: {acc_df[self.target_names].mean().apply(lambda x: round(x, 2)).to_dict()}") #! round(2) fails

    def write_metrics(self, writer, metrics, epoch, mode='val'):
        err_df, acc_df = metrics['err'], metrics['acc']
        for label in self.target_names:
            writer.add_scalar(f'accuracy/{mode}/{label}', acc_df[label].mean(), epoch)
            writer.add_scalar(f'loss/{mode}/{label}', err_df[label].mean(), epoch)

    def calc_metrics(self, labels, predictions):
        return {
            'err': self.err_metric(labels, predictions),
            'acc': self.acc_metric(self.err_metric(labels, predictions))
        }

    def err_metric(self, labels, predictions):
        err_df = pd.DataFrame(columns=self.label_names)
        for label_name in filter(None, self.target_names):
            if label_name not in self.periodic_names:
                abs_err = np.abs(labels[label_name] - predictions[label_name])
            else:
                diff = (labels[label_name] - predictions[label_name]) * np.pi / 180
                abs_err = np.abs(np.arctan2(np.sin(diff), np.cos(diff))) * 180 / np.pi
            err_df[label_name] = abs_err
        return err_df

    def acc_metric(self, err_df):
        """ Accuracy metric for regression problem, counts number of predictions within a tolerance. """
        acc_df = pd.DataFrame(columns=[*self.label_names, 'overall_acc'])
        for label_name, tolerance in zip(filter(None, self.target_names), self.tolerances):
            acc_df[label_name] = (err_df[label_name] < tolerance).astype(np.float32)
        acc_df['overall_acc'] = acc_df[list(filter(None, self.target_names))].all(axis=1).astype(np.float32)
        return acc_df


class LabelledModel:
    def __init__(self, model, image_processing_params, labeller, ros2_topic=None, device='cpu'):
        self.model, self.labeller = model, labeller
        self.image_proc_params = image_processing_params
        self.label_names, self.target_names = labeller.label_names, labeller.target_names
        self.device, self.ros2_topic = device, ros2_topic

    def predict(self, tactile_image):

        proc_image = transform_image(tactile_image, **self.image_proc_params) # gray=True,

        proc_image = np.expand_dims(proc_image.transpose(2, 0, 1), axis=0) # Transpose HWC to CHW and prefix batch dim

        outputs = self.model(torch.from_numpy(proc_image).float().to(self.device))

        if self.model.type.startswith(('MDN', 'GDN')):
            pred_dict = self.labeller.decode(outputs[1].squeeze())
        else:
            pred_dict = self.labeller.decode(outputs)

        if self.ros2_topic is not None:
            self.ros2_topic(pred_dict)

        return np.array([pred_dict[label].item() if label in pred_dict else 0 for label in self.label_names])
