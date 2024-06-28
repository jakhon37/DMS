import argparse
import torch
import os

from numpy import pi as PI  # Constants
from torch import nn
from model import SixDRepNet


def pure_torch_atan2(y, x):
    # Improved version of:
    # https://gist.github.com/nikola-j/b5bb6b141b8d9920318677e1bba70466
    
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * PI
    ans -= ((y < 0) & (x < 0)) * PI
    ans *= (1 - ((y > 0) & (x == 0)) * 1.0)
    ans += ((y > 0) & (x == 0)) * (PI / 2)
    ans *= (1 - ((y < 0) & (x == 0)) * 1.0)
    ans += ((y < 0) & (x == 0)) * (-PI / 2)
    return ans


class ComputeEulerAngles(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, R):
        # Improved version of:
        # utils.compute_euler_angles_from_rotation_matrices

        sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
        singular = sy<1e-6
        singular = singular.float()
            
        x = pure_torch_atan2(R[:,2,1], R[:,2,2])
        y = pure_torch_atan2(-R[:,2,0], sy)
        z = pure_torch_atan2(R[:,1,0],R[:,0,0])
        
        xs = pure_torch_atan2(-R[:,1,2], R[:,1,1])
        ys = pure_torch_atan2(-R[:,2,0], sy)
        zs = R[:,1,0]*0
        
        pitch = x*(1-singular)+xs*singular
        yaw = y*(1-singular)+ys*singular
        roll = z*(1-singular)+zs*singular

        R_pred = torch.stack([pitch, yaw, roll], dim=-1).reshape(-1, 3)
        return R_pred * 180 / PI

#
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Export ONNX.')
    parser.add_argument('--snapshot',
                        dest='/home/fssv2/mushtariy/Head_pose/6DRepNet/sixdrepnet/output/sixdrepnet_model98bgr.pth', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--onnx-path',
                        dest='/home/fssv2/mushtariy/Head_pose/6DRepNet/sixdrepnet/out_model_onnx/out.onnx', help='ONNX save path.',
                        type=str, required=True)
    parser.add_argument('--no-angle-compute',
                        action='store_true', help='Do not apply angle compute module')

    args = parser.parse_args()
    return args


def main(args):
    snapshot_path = args.snapshot
    onnx_output_path = args.onnx_path
    no_angle_compute = args.no_angle_compute

    print(
        'Loading libraries ...'
    )
    import torch
    import onnx
    from onnxsim import simplify

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
    
    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)

    model.eval()

    if not no_angle_compute:
        # Attach final postprocessing layers
        model = nn.Sequential(
            model,
            ComputeEulerAngles(),
        )

    dummy_input = torch.randn(1, 3, 224, 224)

    output_names = ['output']
    dynamic_axes = {
        'images': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        },
    }

    print(f'Output names: {", ".join(output_names)}')
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_output_path,
        verbose=False,
        opset_version=14,  # default, change if required\
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print('Reading exported onnx model ...')
    onnx_model = onnx.load(onnx_output_path)

    print('Trying to simplify ONNX model architecture ...')
    onnx_model, ckeck_ok = simplify(onnx_model)

    if ckeck_ok:
        onnx.save(onnx_model, onnx_output_path)
        print(f"Done simplifying model. Overwrote {onnx_output_path}")
    else:
        print(
            f"Failed to simplify using onnxsim, just use model as-is: {onnx_output_path}"
        )

if __name__ == '__main__':
    args = parse_args()
    main(args)




       
       
        
         
          
           