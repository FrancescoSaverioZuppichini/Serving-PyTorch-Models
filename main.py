import torch
from PIL import Image
import torch.nn as nn
from torchvision.models import alexnet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
# from mlflow.pytorch import _PyTorchWrapper
from mlflow.pyfunc import save_model, PythonModel

# Define the model class
import mlflow.pyfunc


class MyPyTorchWrapper(PythonModel):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def predict(self, data, device='cpu'):
        import torch
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data should be pandas.DataFrame")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(
                np.array(data['x']).astype(np.float32)).to(device)
            preds = self.pytorch_model(input_tensor)
            if not isinstance(preds, torch.Tensor):
                raise TypeError("Expected PyTorch model to output a single output tensor, "
                                "but got output of type '{}'".format(type(preds)))
            predicted = pd.DataFrame(preds.numpy())
            return predicted


model = alexnet(True)
model.eval()

wrapper = MyPyTorchWrapper(model)

save_model(python_model=wrapper, path='alexnet')


"""
channels:
  - defaults
  - pytorch
  dependencies:
  - python=3.8.1
  - pytorch=1.4.0
  - torchvision=0.5.0
  - pip:
    - mlflow
    - cloudpickle==1.3.0
  name: mlflow-env
"""