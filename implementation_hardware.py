import os
import sys
import json
import platform
import importlib.metadata as metadata
from pathlib import Path
from datetime import datetime


def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return 'Not installed / unavailable'


def fmt_gb(n):
    return f'{n / (1024**3):.2f} GB' if n is not None else 'Unavailable'


info = {
    'date': datetime.now().astimezone().isoformat(timespec='seconds'),
    'os': platform.platform(),
    'architecture': platform.machine(),
    'python': platform.python_version(),
    'cpu': platform.processor() or 'Not reported by OS',
    'logical_cores': os.cpu_count(),
    'physical_cores': None,
    'ram_total': None,
    'ram_available': None,
    'torch': version('torch'),
    'numpy': version('numpy'),
    'pandas': version('pandas'),
    'scikit_learn': version('scikit-learn'),
    'statsmodels': version('statsmodels'),
    'cuda_available': False,
    'cuda_version': None,
    'cudnn_version': None,
    'gpu_devices': [],
    'device_used_by_lstm_code': 'CPU',
}

try:
    import psutil
    info['physical_cores'] = psutil.cpu_count(logical=False)
    mem = psutil.virtual_memory()
    info['ram_total'] = fmt_gb(mem.total)
    info['ram_available'] = fmt_gb(mem.available)
except ImportError:
    pass

try:
    import torch
    info['cuda_available'] = torch.cuda.is_available()
    info['cuda_version'] = torch.version.cuda
    info['cudnn_version'] = torch.backends.cudnn.version()
    if torch.cuda.is_available():
        info['device_used_by_lstm_code'] = 'CUDA GPU'
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['gpu_devices'].append({
                'index': i,
                'name': props.name,
                'memory': fmt_gb(props.total_memory),
            })
except ImportError:
    pass

lines = [
    'IMPLEMENTATION AND HARDWARE',
    '=' * 40,
    f'Generated: {info["date"]}',
    f'Operating system: {info["os"]}',
    f'Architecture: {info["architecture"]}',
    f'CPU: {info["cpu"]}',
    f'CPU cores (physical/logical): {info["physical_cores"]} / {info["logical_cores"]}',
    f'RAM (total / currently available): {info["ram_total"]} / {info["ram_available"]}',
    f'Python: {info["python"]}',
    f'PyTorch: {info["torch"]}',
    f'NumPy: {info["numpy"]}',
    f'Pandas: {info["pandas"]}',
    f'Scikit-learn: {info["scikit_learn"]}',
    f'Statsmodels: {info["statsmodels"]}',
    f'CUDA available: {info["cuda_available"]}',
    f'CUDA version reported by PyTorch: {info["cuda_version"]}',
    f'cuDNN version: {info["cudnn_version"]}',
    f'LSTM code execution device (cuda if available, else cpu): {info["device_used_by_lstm_code"]}',
]
for gpu in info['gpu_devices']:
    lines.append(f'GPU {gpu["index"]}: {gpu["name"]} ({gpu["memory"]})')
if not info['gpu_devices']:
    lines.append('GPU used for LSTM: None (CPU execution)')

lines += [
    '',
    'IMPLEMENTATION DETAILS FROM YOUR SHARED CODE',
    'Python; PyTorch (LSTM); Adam optimizer; MinMaxScaler (scikit-learn).',
    'Grid search for hyperparameters; chronological train/validation/test split.',
    'One-step-ahead validation and test with observed previous values.',
    'These are details of the code discussed; verify that your current notebook uses them.',
    '',
    'PAPER DRAFT (complete with values above)',
    'The models were implemented in Python [version]. LSTM models were developed '
    'using PyTorch [version], with NumPy [version], pandas [version], and '
    'scikit-learn [version] for data processing and evaluation. Experiments '
    'were executed on [CPU model] with [RAM] RAM, using [GPU, if applicable] '
    'and [OS].',
]

output = Path.cwd() / 'implementation_hardware.txt'
output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print(f'\nSaved to: {output.resolve()}')
