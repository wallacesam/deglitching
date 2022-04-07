import os
from setuptools import setup, find_packages

# retrieving version
f   = open(os.path.join('deglitching', 'version'), 'r')
ver = f.readline()
f.close()

setup(
    name        = 'deglitching',
    version     = ver,
    description = 'Python library to deglitch Larch XAS data',
    author      = 'Samuel M. Wallace',
    url         = 'https://github.com/wallacesam/deglitching',
    license     = 'BSD',
    test_suite  = 'tests',
    packages    = find_packages(),
    long_description = open('README.md').read(),
    include_package_data = True,
    install_requires = [
        'xraylarch>=0.9.47',
        'numpy>=1.16.4',
        'scipy>=1.3.1',
        'bokeh>=2.2.3',
        'nodejs>=15.11.0',
        'ipywidgets>=7.5.1',
    ],
)