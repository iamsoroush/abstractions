from setuptools import setup

with open("model/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='echomodel',
    version='0.0.1',
    description="EchoModel code for model-creation and inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aimedic.co/soroush.moazed/echotrain",
    packages=['model'],
    python_requires=">=3.6"
)