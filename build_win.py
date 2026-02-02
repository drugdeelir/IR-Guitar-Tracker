
import PyInstaller.__main__
import sys
import os

def build_exe():
    # Define the main script
    script = 'main.py'

    # Define assets to include
    # Format: (source, destination)
    # On Windows, PyInstaller expects ';' as separator for data files, but we can pass them as a list to the API
    assets = [
        ('style.qss', '.'),
        ('logo.png', '.'),
        ('HELP_MAINSTAGE.md', '.'),
    ]

    # Check if logo.mkv exists and add it if it does
    if os.path.exists('logo.mkv'):
        assets.append(('logo.mkv', '.'))

    # Base flags
    args = [
        script,
        '--name=ProjectionMapper',
        '--onefile',      # Bundle everything into a single .exe
        '--windowed',     # Don't show console on launch
        '--noconfirm',    # Overwrite output directory without asking
        '--clean',        # Clean PyInstaller cache
    ]

    # Add assets
    for src, dest in assets:
        # Use platform specific separator for data files
        args.extend(['--add-data', f'{src}{os.pathsep}{dest}'])

    # Add icon if it exists (for Windows, should be .ico, but we can try .icns if that's what we have or just skip)
    if os.path.exists('app.ico'):
        args.extend(['--icon=app.ico'])
    elif os.path.exists('logo.png'):
        # Note: PyInstaller usually needs .ico for Windows icons
        pass

    print(f"Building executable with args: {' '.join(args)}")
    PyInstaller.__main__.run(args)

if __name__ == '__main__':
    build_exe()
