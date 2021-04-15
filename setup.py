import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="probcox",
    version="0.0.4",
    author="Alexander Wolfgang Jung",
    author_email="alex.w.jung@googlemail.com",
    description="Probilistic Cox Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexwjung/ProbCox",
    project_urls={
        "Bug Tracker": "https://github.com/alexwjung/ProbCox/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy", "torch", "pyro-ppl<1.6"],
    python_requires=">=3.6",
)
