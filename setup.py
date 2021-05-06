#!/usr/bin/env python

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

try:
    from torchsynth import __info__ as info
except ImportError:
    # alternative https://stackoverflow.com/a/67692/4521646
    sys.path.append("torchsynth")
    import __info__ as info

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="torchsynth",
    version=info.__version__,
    description=info.__docs__,
    author=info.__author__,
    author_email=info.__author_email__,
    url=info.__homepage__,
    download_url="https://github.com/torchsynth/torchsynth/",
    license=info.__license__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/torchsynth/torchsynth/issues",
        "Documentation": "https://torchsynth.rtfd.io/en/latest/",
        "Source Code": "https://github.com/torchsynth/torchsynth/",
    },
    packages=find_packages(exclude=("tests", "examples")),
    package_data={"torchsynth": ["nebulae/voice/*.json"]},
    # package_dir={"": "src"},
    entry_points={"console_scripts": ["torchsynth.profile=torchsynth.profile:main"]},
    # python_requires=">=3.6",
    python_requires=">=3.4",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "pytorch-lightning",
        # pypi release (only master) doesn't support OrderedDict typing
        # "typing-extensions",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "nbstripout==0.3.9",  # Used in precommit hooks
            "black==20.8b1",  # Used in precommit hooks
            "jupytext==v1.10.3",  # Used in precommit hooks
            "pytest",
            "pytest-cov",
            "ipython",
            "librosa",
            "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
            "matplotlib",
            "numba>=0.49.0",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "pytest-env",
            "sphinx>=3.0.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "unofficial-pt-lightning-sphinx-theme",
            # Temporarily disabled so we can push to pypi
            # "pt-lightning-sphinx-theme @ https://github.com/PyTorchLightning/lightning_sphinx_theme/tarball/master#egg=pt-lightning-sphinx-theme",
            "sphinxcontrib-napoleon",
            "sphinx-autodoc-typehints",
            "mock",
            "sphinx_rtd_theme",
            "myst_parser",
            "linkify-it-py",
        ],
    },
    classifiers=[
        "Environment :: Console",
        "Environment :: GPU",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
