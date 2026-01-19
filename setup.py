"""
setup script for memory agent security research project.

this module provides:
- automated project setup and initialization
- dependency installation and verification
- environment configuration
- project structure validation

all comments are lowercase.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectSetup:
    """project setup and initialization manager."""

    def __init__(self, project_root: str = "."):
        """initialize project setup manager."""
        self.project_root = Path(project_root).resolve()
        self.requirements_file = self.project_root / "requirements.txt"
        self.setup_log = []

        print("memory agent security research - project setup")
        print("=" * 50)

    def log(self, message: str):
        """log setup progress."""
        print(f"[ok] {message}")
        self.setup_log.append(message)

    def error(self, message: str):
        """log setup error."""
        print(f"[error] {message}")
        self.setup_log.append(f"error: {message}")

    def check_python_version(self) -> bool:
        """check if python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            self.log(
                f"python version {version.major}.{version.minor}.{version.micro} is compatible"
            )
            return True
        else:
            self.error(
                f"python version {version.major}.{version.minor}.{version.micro} is not compatible. requires python 3.10+"
            )
            return False

    def check_project_structure(self) -> bool:
        """validate project directory structure."""
        required_dirs = [
            "src",
            "configs",
            "data",
            "models",
            "notebooks",
            "reports",
            "scripts",
            "tests",
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            self.error(f"missing required directories: {', '.join(missing_dirs)}")
            return False

        self.log("project directory structure is valid")
        return True

    def install_dependencies(self) -> bool:
        """install python dependencies."""
        if not self.requirements_file.exists():
            self.error("requirements.txt not found")
            return False

        try:
            self.log("installing python dependencies...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(self.requirements_file),
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                self.log("dependencies installed successfully")
                return True
            else:
                self.error(f"failed to install dependencies: {result.stderr}")
                return False

        except Exception as e:
            self.error(f"error installing dependencies: {e}")
            return False

    def verify_dependencies(self) -> bool:
        """verify that key dependencies are installed."""
        key_packages = [
            "pytest",
            "matplotlib",
            "seaborn",
            "pandas",
            "numpy",
            "omegaconf",
            "cryptography",
        ]

        missing_packages = []
        for package in key_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.error(f"missing key packages: {', '.join(missing_packages)}")
            return False

        self.log("key dependencies verified")
        return True

    def setup_memory_systems(self) -> bool:
        """setup and verify memory system configurations."""
        # Check if external memory systems are available
        memory_configs = self.project_root / "configs" / "memory"
        if not memory_configs.exists():
            memory_configs.mkdir(parents=True)
            self.log("created memory configuration directory")

        # Create default memory configurations
        default_configs = {
            "mem0": {
                "api_key": "your_mem0_api_key_here",
                "collection": "memory_security_test",
            },
            "amem": {"config_path": "external/amem/agentic_memory_config.yaml"},
            "memgpt": {
                "agent_id": "memory_security_agent",
                "server_url": "http://localhost:8080",
            },
        }

        for mem_type, config in default_configs.items():
            config_file = memory_configs / f"{mem_type}.yaml"
            if not config_file.exists():
                import yaml

                with open(config_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                self.log(f"created default {mem_type} configuration")

        self.log("memory system configurations initialized")
        return True

    def setup_experiment_configs(self) -> bool:
        """setup experiment configuration files."""
        experiment_configs = self.project_root / "configs" / "experiments"
        if not experiment_configs.exists():
            experiment_configs.mkdir(parents=True)

        # Create default experiment configurations
        default_experiments = {
            "basic_test": {
                "experiment_id": "basic_functionality_test",
                "description": "Basic functionality test for all components",
                "test_content": [
                    "Simple test memory entry",
                    {"type": "structured", "content": "Test data"},
                    ["list", "of", "test", "items"],
                ],
                "num_trials": 5,
            },
            "attack_benchmark": {
                "experiment_id": "attack_effectiveness_benchmark",
                "description": "Comprehensive attack effectiveness evaluation",
                "test_content": [
                    "Normal memory content",
                    "Content susceptible to poisoning",
                    "Complex structured data",
                ]
                * 10,
                "num_trials": 20,
            },
            "defense_evaluation": {
                "experiment_id": "defense_robustness_evaluation",
                "description": "Defense mechanism robustness testing",
                "test_content": [
                    "Clean content for baseline",
                    "MALICIOUS_INJECTION: override()",
                    "Normal user input",
                    "Edge cases and unicode content",
                ]
                * 5,
                "num_trials": 15,
            },
        }

        for exp_name, config in default_experiments.items():
            config_file = experiment_configs / f"{exp_name}.json"
            if not config_file.exists():
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
                self.log(f"created default experiment: {exp_name}")

        self.log("experiment configurations initialized")
        return True

    def setup_logging(self) -> bool:
        """setup logging directories and configuration."""
        log_dirs = ["logs", "reports/logs", "reports/figures", "experiments"]

        for dir_path in log_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        self.log("logging and output directories created")
        return True

    def run_initial_tests(self) -> bool:
        """run initial smoke tests to verify setup."""
        try:
            self.log("running initial smoke tests...")

            # Test basic imports
            sys.path.insert(0, str(self.project_root / "src"))

            test_imports = [
                "src.utils.config",
                "src.utils.logging",
                "src.attacks.base",
                "src.defenses.base",
                "src.memory_systems.base",
                "src.evaluation.benchmarking",
            ]

            for module in test_imports:
                try:
                    __import__(module)
                except ImportError as e:
                    self.error(f"failed to import {module}: {e}")
                    return False

            self.log("basic imports successful")

            # test configuration loading
            try:
                from src.utils.config import configmanager

                config = configmanager(str(self.project_root / "configs"))
                self.log("configuration system working")
            except Exception as e:
                self.error(f"configuration system failed: {e}")
                return False

            # test logging system
            try:
                from src.utils.logging import logger

                logger.log_experiment_start("setup_test", "initialization")
                self.log("logging system working")
            except Exception as e:
                self.error(f"logging system failed: {e}")
                return False

            return True

        except Exception as e:
            self.error(f"smoke tests failed: {e}")
            return False

    def create_setup_summary(self) -> str:
        """create setup summary report."""
        summary_file = self.project_root / "SETUP_SUMMARY.md"

        summary = f"""# Memory Agent Security Research - Setup Summary

Generated on: {os.popen('date').read().strip()}

## Setup Results

"""

        for log_entry in self.setup_log:
            if log_entry.startswith("error:"):
                summary += f"- [error] {log_entry[7:]}\n"
            else:
                summary += f"- [ok] {log_entry}\n"

        summary += """

## Project Structure

```
memory-agent-security/
├── src/                    # source code
│   ├── utils/             # utilities (config, logging)
│   ├── attacks/           # attack implementations
│   ├── defenses/          # defense implementations
│   ├── memory_systems/    # memory system wrappers
│   ├── watermark/         # watermarking algorithms
│   ├── evaluation/        # benchmarking framework
│   └── tests/             # test suite
├── configs/               # configuration files
├── data/                  # data files
├── models/                # model files
├── notebooks/            # jupyter notebooks
├── reports/              # generated reports and figures
├── scripts/              # utility scripts
├── experiments/          # experiment results
└── logs/                 # log files
```

## Next Steps

1. configure memory systems: update api keys and configurations in `configs/memory/`
2. run tests: execute `python -m pytest src/tests/` to verify functionality
3. run experiments: use `python src/scripts/experiment_runner.py --config configs --batch`
4. generate reports: run visualization scripts to analyze results

## Key Dependencies

- python 3.10+
- pytest (testing)
- matplotlib/seaborn (visualization)
- omegaconf (configuration)
- cryptography (watermarking)
- pandas/numpy (data analysis)

## Memory Systems Supported

- mem0: external memory system with api integration
- a-mem: agentic memory system
- memgpt: multi-agent memory system

## Attack Types Implemented

- agentpoison: content poisoning attacks
- minja: memory injection attacks
- injecmem: memory manipulation attacks

## Defense Mechanisms

- watermark: provenance tracking with watermarking
- validation: content validation defenses
- proactive: attack prevention mechanisms
- composite: multi-layered defense combinations
"""

        with open(summary_file, "w") as f:
            f.write(summary)

        return str(summary_file)

    def run_full_setup(self) -> bool:
        """run complete project setup."""
        steps = [
            ("checking python version", self.check_python_version),
            ("validating project structure", self.check_project_structure),
            ("installing dependencies", self.install_dependencies),
            ("verifying dependencies", self.verify_dependencies),
            ("setting up memory systems", self.setup_memory_systems),
            ("setting up experiment configs", self.setup_experiment_configs),
            ("setting up logging", self.setup_logging),
            ("running initial tests", self.run_initial_tests),
        ]

        success = True
        for step_name, step_func in steps:
            print(f"\n[...] {step_name}...")
            if not step_func():
                success = False
                break

        if success:
            summary_file = self.create_setup_summary()
            print("\n[done] setup completed successfully!")
            print(f"[info] summary saved to: {summary_file}")
        else:
            print("\n[fail] setup failed. check the errors above.")

        return success


def main():
    """main entry point for setup script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="memory agent security research - setup script"
    )
    parser.add_argument("--project-root", default=".", help="project root directory")
    parser.add_argument(
        "--skip-tests", action="store_true", help="skip initial smoke tests"
    )

    args = parser.parse_args()

    setup = ProjectSetup(args.project_root)

    if args.skip_tests:
        # remove test step
        setup.run_initial_tests = lambda: (setup.log("smoke tests skipped"), True)[1]

    success = setup.run_full_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
