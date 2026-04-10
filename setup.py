from setuptools import setup, find_packages

setup(
    name="postex-agent",
    version="1.0.0",
    description="RL-Guided Linux Post-Exploitation Decision Engine",
    author="PostEx Agent Team",
    packages=find_packages(include=["postex_agent", "postex_agent.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "paramiko>=3.0.0",
        "pymetasploit3>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "postex-agent=postex_agent.cli.agent_cli:main",
        ]
    },
    python_requires=">=3.8",
)
