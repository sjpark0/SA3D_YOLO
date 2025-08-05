import os
from torch.utils.cpp_extension import load
from torch_efficient_distloss import eff_distloss


parent_dir = os.path.dirname(os.path.abspath(__file__))
ub360_utils_cuda = load(
        name='ub360_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/ub360_utils.cpp', 'cuda/ub360_utils_kernel.cu']],
        verbose=True)
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

sources=['cuda/adam_upd.cpp', 'cuda/adam_upd_kernel.cu']
adam_upd_cuda = load(
        name='adam_upd_cuda',
        sources=[os.path.join(parent_dir, path) for path in sources],
        verbose=True)

parent_dir = '/opt/conda/lib/python3.10/site-packages/torch_efficient_distloss'
sources = [
        os.path.join(parent_dir, path)
        for path in ['cuda/segment_cumsum.cpp', 'cuda/segment_cumsum_kernel.cu']]
segment_cumsum_cuda = load(
                name='segment_cumsum_cuda',
                sources=sources,
                verbose=True)