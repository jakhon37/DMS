import argparse
import torch
import torch.nn as nn
import onnx
import onnxsim
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'drepnet')))
from model6d import SixDRepNet

class HeadPoseModelWrapper(nn.Module):
    def __init__(self, base_model):
        super(HeadPoseModelWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        rotation_matrices = self.base_model(x)
        euler_angles = self.compute_euler_angles(rotation_matrices)
        return euler_angles

    @staticmethod
    def compute_euler_angles(rotation_matrices):
        batch = rotation_matrices.shape[0]
        R = rotation_matrices
        sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
        singular = sy < 1e-6

        x = HeadPoseModelWrapper.custom_atan2(R[:, 2, 1], R[:, 2, 2])
        y = HeadPoseModelWrapper.custom_atan2(-R[:, 2, 0], sy)
        z = HeadPoseModelWrapper.custom_atan2(R[:, 1, 0], R[:, 0, 0])

        xs = HeadPoseModelWrapper.custom_atan2(-R[:, 1, 2], R[:, 1, 1])
        ys = HeadPoseModelWrapper.custom_atan2(-R[:, 2, 0], sy)
        zs = torch.zeros_like(z)

        euler_angles = torch.zeros(batch, 3, device=rotation_matrices.device)
        euler_angles[:, 0] = torch.where(singular, xs, x)
        euler_angles[:, 1] = torch.where(singular, ys, y)
        euler_angles[:, 2] = torch.where(singular, zs, z)

        return euler_angles

    @staticmethod
    def custom_atan2(y, x):
        # Calculate arctangent using available operations
        pi = torch.tensor(np.pi, device=y.device)
        atan = torch.atan(y / (x + 1e-7))  # Small epsilon to avoid division by zero
        atan2 = torch.where(x > 0, atan, torch.where(y >= 0, atan + pi, atan - pi))
        atan2 = torch.where((x < 0) & (y < 0), atan - pi, atan2)
        atan2 = torch.where((x < 0) & (y >= 0), atan + pi, atan2)
        return atan2

def main(args):
    # Initialize the base model
    base_model = SixDRepNet(backbone_name='RepVGG-B1g2',
                            backbone_file='',
                            deploy=True,
                            pretrained=False)
    # Wrap the base model
    model = HeadPoseModelWrapper(base_model)

    # Prepare dummy input for exporting
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input shape as necessary

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, args.output_path, export_params=True,
                      opset_version=12, do_constant_folding=True, input_names=['input'],
                      output_names=['euler_angles'], dynamic_axes={'input': {0: 'batch_size'},
                                                                    'euler_angles': {0: 'batch_size'}})

    # Simplify the ONNX model
    model_onnx = onnx.load(args.output_path)
    # model_simp, check = onnxsim.simplify(model_onnx)

    # if check:
    #     onnx.save(model_simp, args.output_path)
    #     print(f"Simplified ONNX model saved to {args.output_path}")
    # else:
    #     print("Simplification failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export and simplify a Head Pose Model to ONNX format.')
    parser.add_argument('--input_path', type=str, required=True, help='Path of pytorch model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the exported ONNX model.')

    args = parser.parse_args()
    main(args)
