# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['uvvis_kinetics.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['matplotlib.backends.backend_tkagg', 'PIL._tkinter_finder'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib.tests'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UV-Vis Kinetics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UV-Vis Kinetics',
)
app = BUNDLE(
    coll,
    name='UV-Vis Kinetics.app',
    icon=None,
    bundle_identifier='com.local.uvviskinetics',
)
