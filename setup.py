from setuptools import setup, find_packages

setup(
    name="agent_fields",
    version="0.1.0",
    description="Agent Field Theory — measure and compare AI agent behavior",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy"],
)
