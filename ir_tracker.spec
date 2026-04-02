# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for IR Guitar Tracker
# Build with:  pyinstaller ir_tracker.spec
# Output:      dist/IRGuitarTracker/IRGuitarTracker.exe  (onedir) or dist/IRGuitarTracker.exe (onefile)

import os
from pathlib import Path

block_cipher = None
_root = Path(SPECPATH)

a = Analysis(
    ['main.py'],
    pathex=[str(_root)],
    binaries=[],
    datas=[
        (str(_root / 'style.qss'),      '.'),
        (str(_root / 'marker_data.json'), '.'),
        # Assets — include only if they exist at build time
        *([( str(_root / 'logo.png'), '.')] if (_root / 'logo.png').exists() else []),
        *([( str(_root / 'logo.mkv'), '.')] if (_root / 'logo.mkv').exists() else []),
        *([( str(_root / 'icon.ico'), '.')] if (_root / 'icon.ico').exists() else []),
    ],
    hiddenimports=[
        'mido.backends.rtmidi',
        'mido.backends.pygame',
        'cv2',
        'numpy',
    ],
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
    name='IRGuitarTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_root / 'icon.ico') if (_root / 'icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IRGuitarTracker',
)
