# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../kernel/sidecar/app.py'],
    pathex=['../kernel/sidecar'],
    binaries=[],
    datas=[
        ('../packs', 'packs'),
        ('../kernel/sidecar/retrieval', 'retrieval'),
    ],
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'qdrant_client',
        'sentence_transformers',
        'supervision',
        'pymupdf',
        'fitz',
        'docx',
        'fastembed',
        'lancedb',
        'psutil',
        'kernel.sidecar.pdf_fastpath',
        'kernel.sidecar.ingest.quote_align',
        'kernel.sidecar.retrieval.vector_store',
        'kernel.sidecar.retrieval.embeddings',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='kairo-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
