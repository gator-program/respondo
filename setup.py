"""Setup for respondo"""
import os
from setuptools import find_packages, setup


def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None)
    )


def respondo_setup(*args, **kwargs):
    if is_conda_build():
        kwargs.pop("install_requires")
        kwargs.pop("setup_requires")
        kwargs.pop("tests_require")
    try:
        setup(*args, **kwargs)
    except Exception as e:
        raise RuntimeError("respondo setup.py failed.")


respondo_setup(
    name="respondo",
    description="respondo: Library for Response Functions",
    keywords=[
        "electronic", "structure", "computational", "chemistry", "quantum",
        "spectroscopy", "response", "theory", "molecular", "properties",
        "ADC",
    ],
    #
    # author="The Respondo Authors",
    # author_email="",
    # license="GPL v3",
    # url="",
    # project_urls={
    #     "Source": "",
    #     "Issues": "",
    # },
    #
    version="0.0.1",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        # "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "License :: Free For Educational Use",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Education",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    #
    packages=find_packages(exclude=["*.test*", "test_*"]),
    zip_safe=False,
    #
    platforms=["Linux", "Mac OS-X"],
    python_requires=">=3.6",
    install_requires=[
        "numpy >= 1.14",
        "adcc >= 0.15.6",
    ],
    tests_require=["pytest" , "pyscf"],
    #
)
