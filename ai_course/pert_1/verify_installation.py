
import sys
import importlib

packages_to_check = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'scikit-fuzzy',     
    'opencv-python',
    'jupyter',
    'jupyterlab',
    'ipykernel',
    'joblib',
    'nltk',
    'tensorflow',
    'tensorflow-datasets',
    'transformers',
    'fastapi'
]

print("=" * 80)
print(f"Python: {sys.version}")
print("=" * 80)
print("Verifying package installations...\n")

installed = []
failed = []

for pkg in packages_to_check:
    try:
        import_name = pkg
        special_versions = {}
        
        pkg_mappings = {
            'scikit-learn': 'sklearn',
            'opencv-python': 'cv2',
            'scikit-fuzzy': 'skfuzzy',
            'tensorflow-datasets': 'tensorflow_datasets',
            'nltk': 'nltk',
            'transformers': 'transformers',
            'fastapi': 'fastapi',
            'jupyter': 'jupyter_core'
        }
        
        if pkg in pkg_mappings:
            import_name = pkg_mappings[pkg]
            
        if pkg == 'jupyter':
            try:
                import jupyter_core
                version_tuple = getattr(jupyter_core, 'version_info', (0, 0, 0))
                if isinstance(version_tuple, tuple):
                    version_str = '.'.join(map(str, version_tuple[:3]))
                    special_versions[pkg] = f"ok, version = {version_str}"
                else:
                    special_versions[pkg] = f"ok, version = {version_tuple}"
            except (ImportError, AttributeError) as e:
                special_versions[pkg] = "version check failed"

        m = importlib.import_module(import_name)
        version = (f"ok, version = {getattr(m, '__version__', 'unknown')}")
        if version == 'unknown':
            version = getattr(m, 'version', 'unknown')
            if version != 'unknown' and hasattr(version, 'version'):
                version = version.version
        
        display_version = special_versions.get(pkg, version)
        print(f" {pkg.ljust(20)}: {display_version}")
        installed.append(pkg)
    except Exception as e:
        print(f" error {pkg.ljust(20)}: Not installed ({str(e).split('\n')[0]})")
        failed.append(pkg)

print("\n" + "=" * 80)
print(f"Installation Summary:")
print(f" {len(installed)} packages installed successfully")
if failed:
    print(f" {len(failed)} packages failed to import")
    print("\nMissing packages:")
    for pkg in failed:
        print(f" error {pkg}")
    print("\nYou can install missing packages using: pip install package_name")
print("=" * 80)