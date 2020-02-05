import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Poison_Defence_Analytic(PDA)",
    version="1.0.1",
    author="Iman Zabett",
    author_email="iman.zabett@accenture.com",
    description="This package aims to build a robust AI-Model which will be robust against "
                "poison attacks. First we build a AI-Model based on poisonous dataset, "
                "Spacenet dataset has been selected for application."
                "After building poison model, then we detect poison datapoints "
                "and remove them. By having the cleaned dataset, We build a "
                "new and robust AI-Model which will be  based on cleaned dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)