# Memory Agent Security Research - Usage Guide

## Quick Start

### 1. Project Setup

```bash
# Clone or navigate to project directory
cd memory-agent-security

# Run setup script
python setup.py

# Verify installation
python -c "import src.utils.config; print('Setup successful!')"
```

### 2. Basic Usage

```python
from src.utils.config import configmanager
from src.memory_systems.wrappers import create_memory_system
from src.attacks.implementations import create_attack
from src.defenses.implementations import create_defense

# Initialize configuration
config = configmanager("configs")

# Create memory system (example with mock)
memory = create_memory_system("mem0", {"user_id": "test"})

# Create attack and defense
attack = create_attack("agent_poison")
defense = create_defense("watermark")

# Activate defense
defense.activate()

# Execute attack
content = "Test memory content"
attack_result = attack.execute(content)

# Test defense detection
if attack_result.get("success"):
    detection = defense.detect_attack(attack_result.get("poisoned_content"))
    print(f"Attack detected: {detection['attack_detected']}")

# Cleanup
defense.deactivate()
```

## Running Experiments

### Automated Experiment Runner

```bash
# Run default experiment suite
python src/scripts/experiment_runner.py --config configs --batch

# Run single experiment
python src/scripts/experiment_runner.py --config configs --experiment configs/experiments/basic_test.json

# Generate dashboard
python src/scripts/experiment_runner.py --config configs --batch --dashboard
```

### Custom Experiment Configuration

Create `configs/experiments/custom_experiment.json`:

```json
{
  "experiment_id": "custom_attack_test",
  "description": "Test custom attack scenarios",
  "test_content": [
    "Normal memory content",
    "Content susceptible to attacks",
    { "type": "structured", "data": "complex content" },
    ["array", "of", "memory", "items"]
  ],
  "num_trials": 25
}
```

### Programmatic Experiment Execution

```python
from src.scripts.experiment_runner import ExperimentRunner

# Initialize runner
runner = ExperimentRunner("configs", "experiments")

# Define custom experiment
experiment_config = {
    "experiment_id": "programmatic_test",
    "test_content": ["Test content"] * 10,
    "num_trials": 15
}

# Run experiment
result = runner.run_single_experiment(experiment_config)

# Generate report
report_path = runner.generate_experiment_report([result])
print(f"Report: {report_path}")
```

## Testing Framework

### Running Tests

```bash
# Run all tests
python -m pytest src/tests/ --verbose

# Run specific test categories
python -m pytest src/tests/test_memory_security.py -k "attack"
python -m pytest src/tests/ -m "integration"

# Run with coverage
python -m pytest src/tests/ --cov=src --cov-report=html

# Skip slow tests
python -m pytest src/tests/ -m "not slow"
```

### Test Structure

```
src/tests/
├── conftest.py              # Test configuration and fixtures
├── test_memory_security.py  # Comprehensive test suite
└── pytest.ini              # Pytest configuration
```

### Writing Custom Tests

```python
import pytest
from src.attacks.implementations import create_attack

def test_custom_attack():
    """Test custom attack implementation."""
    attack = create_attack("agent_poison")

    # Test normal execution
    result = attack.execute("test content")
    assert result["attack_type"] == "agent_poison"
    assert "success" in result

    # Test edge cases
    result_empty = attack.execute("")
    assert result_empty["success"] is False

    result_unicode = attack.execute("测试内容")
    assert isinstance(result_unicode, dict)
```

## Visualization and Analysis

### Generating Visualizations

```python
from src.scripts.visualization import BenchmarkVisualizer
from src.evaluation.benchmarking import BenchmarkRunner

# Run benchmark
runner = BenchmarkRunner(config, logger)
results = runner.run_benchmark("viz_test", ["test content"], num_trials=10)

# Create visualizations
visualizer = BenchmarkVisualizer("reports/figures")
saved_plots = visualizer.generate_comprehensive_report([results])

print("Generated plots:", saved_plots)
```

### Available Visualizations

1. **Attack Success Rates**: Bar plots showing ASR-R, ASR-A, ASR-T by memory system
2. **Defense Effectiveness**: TPR vs FPR plots and performance comparisons
3. **Performance Analysis**: Execution time and memory operation tracking
4. **Attack-Defense Heatmap**: Effectiveness matrix for all combinations

### Statistical Analysis

```python
from src.scripts.visualization import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze attack patterns
attack_analysis = analyzer.analyze_attack_patterns(results)
print("Most effective attack:", attack_analysis['summary']['most_effective_attack'])

# Analyze defense robustness
defense_analysis = analyzer.analyze_defense_robustness(results)
print("Most robust defense:", defense_analysis['summary']['most_robust_defense'])
```

## Configuration Management

### Configuration Files Structure

```
configs/
├── memory/           # Memory system configurations
│   ├── mem0.yaml
│   ├── amem.yaml
│   └── memgpt.yaml
└── experiments/      # Experiment configurations
    ├── basic_test.json
    └── attack_benchmark.json
```

### Memory System Configuration

**Mem0 Configuration** (`configs/memory/mem0.yaml`):

```yaml
api_key: "your_mem0_api_key_here"
collection: "memory_security_research"
host: "https://api.mem0.ai"
timeout: 30
retries: 3
```

**A-MEM Configuration** (`configs/memory/amem.yaml`):

```yaml
config_path: "external/amem/agentic_memory_config.yaml"
model_name: "gpt-4"
embedding_model: "text-embedding-ada-002"
```

**MemGPT Configuration** (`configs/memory/memgpt.yaml`):

```yaml
agent_id: "memory_security_agent"
server_url: "http://localhost:8080"
api_key: "your_memgpt_api_key"
model: "gpt-4"
```

### Experiment Configuration

```json
{
  "experiment_id": "comprehensive_evaluation",
  "description": "Full security evaluation",
  "test_content": [
    "Simple text content",
    {"type": "structured", "content": "JSON data"},
    ["list", "of", "items"],
    "Unicode content: 你好世界 🌍",
    "Long content that tests system limits and performance characteristics under various loads and conditions" * 10
  ],
  "num_trials": 50,
  "memory_systems": ["mem0", "amem", "memgpt"],
  "attacks": ["agent_poison", "minja", "injecmem"],
  "defenses": ["watermark", "validation", "proactive", "composite"]
}
```

## Memory System Integration

### Using Real Memory Systems

```python
# Mem0 Integration
mem0_config = {
    "api_key": "your_actual_api_key",
    "collection": "research_test"
}
mem0_system = create_memory_system("mem0", mem0_config)

# Store and retrieve
result = mem0_system.store("Test content", {"source": "test"})
retrieved = mem0_system.retrieve(result["id"])

# A-MEM Integration
amem_config = {
    "config_path": "path/to/amem/config.yaml"
}
amem_system = create_memory_system("amem", amem_config)

# MemGPT Integration
memgpt_config = {
    "agent_id": "research_agent",
    "server_url": "http://localhost:8080"
}
memgpt_system = create_memory_system("memgpt", memgpt_config)
```

### Mock Systems for Testing

```python
# Use mock configurations for testing
mock_config = {"user_id": "test", "mock": True}
mock_memory = create_memory_system("mem0", mock_config)
```

## Attack and Defense Development

### Implementing Custom Attacks

```python
from src.attacks.base import Attack

class CustomAttack(Attack):
    def execute(self, content: Any) -> Dict[str, Any]:
        # Custom attack logic
        modified_content = self._apply_attack(content)

        return {
            "attack_type": "custom_attack",
            "success": True,
            "modified_content": modified_content,
            "timestamp": time.time()
        }

    def _apply_attack(self, content: Any) -> Any:
        # Implement attack logic
        if isinstance(content, str):
            return content.replace("normal", "ATTACKED")
        return content

# Register attack
attack = CustomAttack()
result = attack.execute("This is normal content")
```

### Implementing Custom Defenses

```python
from src.defenses.base import Defense

class CustomDefense(Defense):
    def activate(self) -> bool:
        self.active = True
        return True

    def deactivate(self) -> bool:
        self.active = False
        return True

    def detect_attack(self, content: Any) -> Dict[str, Any]:
        # Detection logic
        is_attack = "ATTACKED" in str(content)

        return {
            "defense_type": "custom_defense",
            "attack_detected": is_attack,
            "confidence": 1.0 if is_attack else 0.0,
            "timestamp": time.time()
        }

# Use defense
defense = CustomDefense()
defense.activate()
result = defense.detect_attack("This content has ATTACKED text")
defense.deactivate()
```

## Performance Optimization

### Memory Management

```python
# Use generators for large datasets
def generate_test_content(num_items: int):
    for i in range(num_items):
        yield f"Test content item {i}"

# Process in batches
batch_size = 100
test_content = list(generate_test_content(1000))

for i in range(0, len(test_content), batch_size):
    batch = test_content[i:i + batch_size]
    # Process batch
    results = runner.run_benchmark(f"batch_{i//batch_size}", batch, num_trials=5)
```

### Parallel Execution

```python
import concurrent.futures

def run_experiment_batch(config):
    runner = ExperimentRunner("configs", "experiments")
    return runner.run_single_experiment(config)

# Run experiments in parallel
configs = create_default_experiment_configs()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_experiment_batch, configs))
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure src is in Python path
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src

   # Or run with proper path
   PYTHONPATH=src python your_script.py
   ```

2. **Configuration Errors**

   ```python
   # Validate configuration
   from src.utils.config import configmanager
   try:
       config = configmanager("configs")
       print("Configuration loaded successfully")
   except Exception as e:
       print(f"Configuration error: {e}")
   ```

3. **Memory System Connection Issues**

   ```python
   # Test connection
   try:
       memory = create_memory_system("mem0", config.get_memory_config("mem0"))
       result = memory.store("test", {})
       print("Connection successful")
   except Exception as e:
       print(f"Connection failed: {e}")
   ```

4. **Test Failures**

   ```bash
   # Run specific failing test with debug output
   python -m pytest src/tests/test_memory_security.py::TestAttacks::test_agent_poison_attack -v -s

   # Check test coverage
   python -m pytest src/tests/ --cov=src --cov-report=term-missing
   ```

### Debug Logging

```python
from src.utils.logging import logger

# Enable debug logging
logger = ResearchLogger("logs", "debug_experiment", console_level="DEBUG")

# Log custom events
logger.log_experiment_start("debug_test", "Debugging session")
logger.log_attack_execution("agent_poison", "test content")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_experiment():
    runner = ExperimentRunner("configs", "experiments")
    config = create_default_experiment_configs()[0]
    result = runner.run_single_experiment(config)
    return result

# Profile execution
profiler = cProfile.Profile()
profiler.enable()
result = profile_experiment()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## Advanced Usage

### Custom Evaluation Metrics

```python
from src.evaluation.benchmarking import AttackEvaluator, DefenseEvaluator

class CustomAttackEvaluator(AttackEvaluator):
    def evaluate_attack(self, attack_type: str, test_content: List[Any], num_trials: int = 10):
        # Custom evaluation logic
        metrics = super().evaluate_attack(attack_type, test_content, num_trials)

        # Add custom metrics
        metrics.custom_score = self._calculate_custom_score(metrics)

        return metrics

    def _calculate_custom_score(self, metrics):
        # Implement custom scoring
        return (metrics.asr_r + metrics.asr_a + metrics.asr_t) / 3

# Use custom evaluator
evaluator = CustomAttackEvaluator()
custom_metrics = evaluator.evaluate_attack("agent_poison", test_content, num_trials=20)
```

### Integration with External Tools

```python
# Weights & Biases integration
import wandb

def log_to_wandb(results):
    wandb.init(project="memory-security-research")

    for result in results:
        wandb.log({
            "experiment_id": result.experiment_id,
            "attack_success_rate": result.attack_metrics["agent_poison"].asr_r,
            "defense_tpr": result.defense_metrics["watermark"].tpr,
            "test_duration": result.test_duration
        })

# Jupyter notebook integration
def create_notebook_report(results):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()

    # Add cells with results
    nb.cells.append(nbf.v4.new_markdown_cell("# Memory Security Research Results"))

    for result in results:
        nb.cells.append(nbf.v4.new_code_cell(f"""
# Experiment: {result.experiment_id}
attack_metrics = {result.attack_metrics}
defense_metrics = {result.defense_metrics}
"""))

    with open("research_report.ipynb", "w") as f:
        nbf.write(nb, f)
```

## Best Practices

### Code Organization

- Keep attack/defense implementations modular
- Use type hints for better code maintainability
- Follow lowercase comment convention
- Add comprehensive docstrings

### Experiment Design

- Use sufficient trial numbers for statistical significance
- Include diverse test content (text, structured data, edge cases)
- Document experiment parameters and expected outcomes
- Save intermediate results for reproducibility

### Performance

- Profile code before optimization
- Use appropriate data structures for large datasets
- Consider memory usage in long-running experiments
- Implement proper cleanup in test fixtures

### Security

- Never commit API keys or sensitive data
- Use environment variables for secrets
- Validate input data in custom implementations
- Be cautious with attack payload logging

## Support and Contributing

### Getting Help

- Check the API reference for detailed documentation
- Review test files for usage examples
- Examine existing implementations for patterns

### Contributing

1. Follow established code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Run full test suite before submitting

### Reporting Issues

- Include full error messages and stack traces
- Provide minimal reproducible examples
- Specify environment details (Python version, OS, dependencies)
- Attach relevant log files and configuration
