# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec file for AI Object Counter Pro.
Packages everything including templates, static assets, model weights, and fonts into a standalone folder.
"""
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all ultralytics and torch data
added_files = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('index.html', '.'),
    ('yolov8n.pt', '.'),
    ('config.py', '.'),
]

# Collect ultralytics assets
try:
    added_files += collect_data_files('ultralytics')
except Exception:
    pass

hidden_imports = [
    'ultralytics',
    'torch',
    'torchvision',
    'cv2',
    'PIL',
    'flask',
    'werkzeug',
    'jinja2',
    'tkinter',
    'tracker_counter',
    'detector',
    'utils.drawing',
]

a = Analysis(
    ['index.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_Object_Counter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True so user can see startup logs
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_Object_Counter',
)
