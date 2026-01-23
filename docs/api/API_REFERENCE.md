# Memory Agent Security Research Framework

## Overview

This framework provides comprehensive tools for researching security vulnerabilities in memory agent systems, including attack characterization, defense development, and systematic evaluation. The framework supports Mem0, A-MEM, and MemGPT memory systems with implementations for AgentPoison, MINJA, and InjecMEM attacks, along with watermarking-based defenses.

## Architecture

### Core Components

#### Memory Systems (`src/memory_systems/`)

- **Base Protocol**: `MemorySystem` protocol defining standard interface
- **Wrappers**: `Mem0Wrapper`, `AMEMWrapper`, `MemGPTWrapper` for external system integration
- **Factory**: `create_memory_system()` for dynamic instantiation

#### Attacks (`src/attacks/`)

- **Base Class**: `Attack` abstract base class with `execute()` method
- **Implementations**:
  - `AgentPoisonAttack`: Content poisoning via character manipulation
  - `MINJAAttack`: Memory injection attacks
  - `InjecMEMAttack`: Memory manipulation attacks
- **Suite**: `AttackSuite` for batch execution

#### Defenses (`src/defenses/`)

- **Base Class**: `Defense` abstract base class with `activate()`, `detect_attack()` methods
- **Implementations**:
  - `WatermarkDefense`: Watermark-based provenance tracking
  - `ContentValidationDefense`: Pattern-based content validation
  - `ProactiveDefense`: Simulation-based attack prevention
  - `CompositeDefense`: Multi-layered defense combination
- **Suite**: `DefenseSuite` for coordinated defense activation

#### Watermarking (`src/watermark/`)

- **Encoders**:
  - `LSBWatermarkEncoder`: Least significant bit steganography
  - `SemanticWatermarkEncoder`: Semantic embedding watermarks
  - `CryptographicWatermarkEncoder`: Cryptographically secure watermarks
  - `CompositeWatermarkEncoder`: Multi-technique combination
- **Tracker**: `ProvenanceTracker` for content origin verification

#### Evaluation (`src/evaluation/`)

- **Metrics**: `AttackMetrics`, `DefenseMetrics` dataclasses
- **Evaluators**: `AttackEvaluator`, `DefenseEvaluator` for performance measurement
- **Runner**: `BenchmarkRunner` for comprehensive experiment execution
- **Reports**: `EvaluationReportGenerator` for result analysis

#### Utilities (`src/utils/`)

- **Configuration**: `configmanager` class with YAML loading and validation
- **Logging**: `ResearchLogger` with file/console output and specialized methods

## API Reference

### Memory Systems

#### MemorySystem Protocol

```python
class MemorySystem(Protocol):
    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store content in memory system."""
        ...

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve content by key."""
        ...

    def search(self, query: Any, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory content."""
        ...

    def get_all_keys(self) -> List[str]:
        """Get all stored memory keys."""
        ...
```

#### Memory System Factory

```python
def create_memory_system(system_type: str, config: Dict[str, Any]) -> MemorySystem:
    """Create memory system instance.

    Args:
        system_type: Type of memory system ('mem0', 'amem', 'memgpt')
        config: Configuration dictionary for the system

    Returns:
        Configured memory system instance

    Raises:
        ValueError: If system_type is unsupported
    """
```

### Attacks

#### Attack Base Class

```python
class Attack(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize attack with configuration."""
        self.config = config or {}

    @abstractmethod
    def execute(self, content: Any) -> Dict[str, Any]:
        """Execute attack on content.

        Args:
            content: Target content to attack

        Returns:
            Dictionary with attack results including:
            - attack_type: Type of attack executed
            - success: Boolean indicating success
            - [attack-specific results]
            - timestamp: Execution timestamp
        """
        pass
```

#### Attack Implementations

##### AgentPoisonAttack

```python
class AgentPoisonAttack(Attack):
    def __init__(self, intensity: float = 0.5, patterns: Optional[List[str]] = None):
        """Initialize poisoning attack.

        Args:
            intensity: Attack intensity (0.0-1.0)
            patterns: Custom poisoning patterns
        """

    def execute(self, content: Any) -> Dict[str, Any]:
        """Execute character-level poisoning."""
```

##### MINJAAttack

```python
class MINJAAttack(Attack):
    def __init__(self, injection_rate: float = 0.3, target_fields: Optional[List[str]] = None):
        """Initialize memory injection attack.

        Args:
            injection_rate: Rate of injection (0.0-1.0)
            target_fields: Fields to target for injection
        """

    def execute(self, content: Any) -> Dict[str, Any]:
        """Execute memory injection attack."""
```

##### InjecMEMAttack

```python
class InjecMEMAttack(Attack):
    def __init__(self, manipulation_level: int = 2, target_positions: Optional[List[int]] = None):
        """Initialize memory manipulation attack.

        Args:
            manipulation_level: Level of manipulation complexity
            target_positions: Positions to target for manipulation
        """

    def execute(self, content: Any) -> Dict[str, Any]:
        """Execute memory manipulation attack."""
```

#### AttackSuite

```python
class AttackSuite:
    def __init__(self, attacks: Optional[List[Attack]] = None):
        """Initialize attack suite.

        Args:
            attacks: List of attack instances to include
        """

    def execute_all(self, content: Any) -> Dict[str, Any]:
        """Execute all attacks on content.

        Args:
            content: Target content

        Returns:
            Dictionary with results from all attacks
        """
```

### Defenses

#### Defense Base Class

```python
class Defense(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize defense with configuration."""
        self.config = config or {}
        self.active = False

    @abstractmethod
    def activate(self) -> bool:
        """Activate defense mechanism.

        Returns:
            True if activation successful
        """
        pass

    @abstractmethod
    def deactivate(self) -> bool:
        """Deactivate defense mechanism.

        Returns:
            True if deactivation successful
        """
        pass

    @abstractmethod
    def detect_attack(self, content: Any) -> Dict[str, Any]:
        """Detect attacks in content.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with detection results:
            - attack_detected: Boolean detection result
            - confidence: Detection confidence (0.0-1.0)
            - [defense-specific results]
            - timestamp: Detection timestamp
        """
        pass
```

#### Defense Implementations

##### WatermarkDefense

```python
class WatermarkDefense(Defense):
    def __init__(self, encoder_type: str = "lsb", tracker: Optional[ProvenanceTracker] = None):
        """Initialize watermark-based defense.

        Args:
            encoder_type: Type of watermark encoder to use
            tracker: Provenance tracker instance
        """
```

##### ContentValidationDefense

```python
class ContentValidationDefense(Defense):
    def __init__(self, strict_mode: bool = False, custom_patterns: Optional[List[str]] = None):
        """Initialize content validation defense.

        Args:
            strict_mode: Enable strict validation mode
            custom_patterns: Additional validation patterns
        """
```

##### ProactiveDefense

```python
class ProactiveDefense(Defense):
    def __init__(self, simulation_depth: int = 3, threshold: float = 0.8):
        """Initialize proactive defense.

        Args:
            simulation_depth: Depth of attack simulation
            threshold: Detection threshold
        """
```

##### CompositeDefense

```python
class CompositeDefense(Defense):
    def __init__(self, defenses: List[Defense], aggregation_method: str = "majority_vote"):
        """Initialize composite defense.

        Args:
            defenses: List of defense instances to combine
            aggregation_method: Method for combining results
        """
```

#### DefenseSuite

```python
class DefenseSuite:
    def __init__(self, defenses: Optional[List[Defense]] = None):
        """Initialize defense suite.

        Args:
            defenses: List of defense instances
        """

    def activate_all(self) -> Dict[str, bool]:
        """Activate all defenses.

        Returns:
            Dictionary mapping defense types to activation results
        """

    def detect_attack(self, content: Any) -> Dict[str, Any]:
        """Run detection across all defenses.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with detection results from all defenses
        """
```

### Watermarking

#### WatermarkEncoder Base Class

```python
class WatermarkEncoder(ABC):
    @abstractmethod
    def embed(self, content: str, watermark: str) -> str:
        """Embed watermark in content.

        Args:
            content: Original content
            watermark: Watermark to embed

        Returns:
            Watermarked content
        """
        pass

    @abstractmethod
    def extract(self, content: str) -> Optional[str]:
        """Extract watermark from content.

        Args:
            content: Watermarked content

        Returns:
            Extracted watermark or None
        """
        pass
```

#### ProvenanceTracker

```python
class ProvenanceTracker:
    def __init__(self, encoder: Optional[WatermarkEncoder] = None):
        """Initialize provenance tracker.

        Args:
            encoder: Watermark encoder to use
        """

    def register_content(self, content_id: str, content: str) -> str:
        """Register content with watermark.

        Args:
            content_id: Unique content identifier
            content: Content to register

        Returns:
            Watermark ID for tracking
        """

    def watermark_content(self, content: str, watermark_id: str) -> str:
        """Apply watermark to content.

        Args:
            content: Content to watermark
            watermark_id: Watermark identifier

        Returns:
            Watermarked content
        """

    def verify_provenance(self, content: str) -> Optional[Dict[str, Any]]:
        """Verify content provenance.

        Args:
            content: Content to verify

        Returns:
            Provenance information or None
        """
```

### Evaluation

#### Metrics Dataclasses

```python
@dataclass
class AttackMetrics:
    attack_type: str
    total_attempts: int
    successful_attempts: int
    asr_r: float  # Retrieval success rate
    asr_a: float  # Availability success rate
    asr_t: float  # Tampering success rate
    execution_time_avg: float

@dataclass
class DefenseMetrics:
    defense_type: str
    total_tests: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    tpr: float  # True positive rate
    fpr: float  # False positive rate
    precision: float
    recall: float
```

#### Evaluators

```python
class AttackEvaluator:
    def evaluate_attack(self, attack_type: str, test_content: List[Any],
                       num_trials: int = 10) -> AttackMetrics:
        """Evaluate attack performance.

        Args:
            attack_type: Type of attack to evaluate
            test_content: List of test content
            num_trials: Number of evaluation trials

        Returns:
            Attack performance metrics
        """

class DefenseEvaluator:
    def evaluate_defense(self, defense_type: str, attack_suite: AttackSuite,
                        clean_content: List[Any], poisoned_content: List[Any]) -> DefenseMetrics:
        """Evaluate defense performance.

        Args:
            defense_type: Type of defense to evaluate
            attack_suite: Attack suite for testing
            clean_content: Clean test content
            poisoned_content: Poisoned test content

        Returns:
            Defense performance metrics
        """
```

#### BenchmarkRunner

```python
class BenchmarkRunner:
    def __init__(self, config: configmanager, logger: ResearchLogger):
        """Initialize benchmark runner.

        Args:
            config: Configuration manager
            logger: Research logger
        """

    def run_benchmark(self, experiment_id: str, test_content: List[Any],
                     num_trials: int = 10) -> BenchmarkResult:
        """Run comprehensive benchmark.

        Args:
            experiment_id: Unique experiment identifier
            test_content: Test content for evaluation
            num_trials: Number of trials per test

        Returns:
            Complete benchmark results
        """
```

### Utilities

#### Configuration Manager

```python
class configmanager:
    def __init__(self, config_dir: str):
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """

    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration by type.

        Args:
            config_type: Type of configuration to load

        Returns:
            Configuration dictionary
        """

    def get_memory_config(self, system_type: str) -> Dict[str, Any]:
        """Get memory system configuration."""

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
```

#### Research Logger

```python
class ResearchLogger:
    def __init__(self, log_dir: str, experiment_id: str, console_level: str = "INFO"):
        """Initialize research logger.

        Args:
            log_dir: Directory for log files
            experiment_id: Current experiment identifier
            console_level: Console logging level
        """

    def log_experiment_start(self, experiment_id: str, description: str):
        """Log experiment start."""

    def log_attack_execution(self, attack_type: str, target_content: Any):
        """Log attack execution."""

    def log_defense_activation(self, defense_type: str):
        """Log defense activation."""

    def log_benchmark_completion(self, experiment_id: str, results: Dict[str, Any]):
        """Log benchmark completion."""
```

## Usage Examples

### Basic Setup

```python
from src.utils.config import configmanager
from src.utils.logging import logger
from src.memory_systems.wrappers import create_memory_system
from src.attacks.implementations import create_attack
from src.defenses.implementations import create_defense

# Initialize components
config = configmanager("configs")
logger = ResearchLogger("logs", "example_experiment")

# Create memory system
memory = create_memory_system("mem0", config.get_memory_config("mem0"))

# Create attack and defense
attack = create_attack("agent_poison")
defense = create_defense("watermark")

# Activate defense
defense.activate()

# Execute attack
test_content = "Test memory content"
attack_result = attack.execute(test_content)

# Test defense detection
if attack_result["success"]:
    detection_result = defense.detect_attack(attack_result["poisoned_content"])
    print(f"Attack detected: {detection_result['attack_detected']}")

# Cleanup
defense.deactivate()
```

### Running Benchmarks

```python
from src.evaluation.benchmarking import BenchmarkRunner

# Initialize benchmark runner
runner = BenchmarkRunner(config, logger)

# Define test content
test_content = [
    "Sample memory entry 1",
    "Sample memory entry 2",
    {"type": "structured", "content": "Test data"}
]

# Run benchmark
results = runner.run_benchmark("comprehensive_test", test_content, num_trials=20)

# Access results
print(f"Attack success rates: {results.attack_metrics}")
print(f"Defense performance: {results.defense_metrics}")
```

### Custom Experiment

```python
from src.scripts.experiment_runner import ExperimentRunner

# Create experiment configuration
experiment_config = {
    "experiment_id": "custom_attack_test",
    "test_content": ["Custom test content"] * 10,
    "num_trials": 15
}

# Run experiment
runner = ExperimentRunner("configs", "experiments")
result = runner.run_single_experiment(experiment_config)

# Generate report
report_path = runner.generate_experiment_report([result])
print(f"Report generated: {report_path}")
```

## Configuration

### Directory Structure

```
configs/
├── memory/
│   ├── mem0.yaml
│   ├── amem.yaml
│   └── memgpt.yaml
└── experiments/
    ├── basic_test.json
    └── attack_benchmark.json
```

### Memory Configuration Example (mem0.yaml)

```yaml
api_key: "your_mem0_api_key"
collection: "memory_security_research"
host: "https://api.mem0.ai"
timeout: 30
```

### Experiment Configuration Example

```json
{
  "experiment_id": "attack_effectiveness_test",
  "description": "Test attack success rates",
  "test_content": ["Normal content", "Susceptible content"],
  "num_trials": 20,
  "memory_systems": ["mem0", "amem"],
  "attacks": ["agent_poison", "minja"],
  "defenses": ["watermark", "validation"]
}
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest src/tests/

# Run specific test categories
python -m pytest src/tests/ -m "not slow"  # Skip slow tests
python -m pytest src/tests/ -k "attack"    # Run attack-related tests

# Run with coverage
python -m pytest src/tests/ --cov=src --cov-report=html
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark execution times and resource usage
- **Smoke Tests**: Basic functionality verification

## Experiment Automation

### Command Line Usage

```bash
# Run single experiment
python src/scripts/experiment_runner.py --config configs --experiment configs/experiments/basic_test.json

# Run batch experiments
python src/scripts/experiment_runner.py --config configs --batch --dashboard

# Custom output directory
python src/scripts/experiment_runner.py --config configs --batch --output my_experiments
```

### Script Integration

```python
from src.scripts.experiment_runner import ExperimentRunner, create_default_experiment_configs

# Run default experiment suite
runner = ExperimentRunner("configs", "experiments")
configs = create_default_experiment_configs()
results = runner.run_batch_experiments(configs)

# Generate visualizations
from src.scripts.visualization import create_experiment_dashboard
dashboard_path = create_experiment_dashboard(results)
```

## Visualization and Analysis

### Generating Reports

```python
from src.scripts.visualization import BenchmarkVisualizer, StatisticalAnalyzer

# Create visualizer
visualizer = BenchmarkVisualizer("reports/figures")

# Generate all plots
saved_plots = visualizer.generate_comprehensive_report(results)

# Statistical analysis
analyzer = StatisticalAnalyzer()
stats_report = analyzer.generate_statistical_report(results, "reports/statistics.json")
```

### Available Visualizations

- **Attack Success Rates**: Bar plots of ASR metrics by memory system and attack type
- **Defense Effectiveness**: TPR/FPR scatter plots and performance comparisons
- **Performance Analysis**: Execution time and memory operation tracking
- **Attack-Defense Heatmap**: Effectiveness matrix visualization

## Error Handling

The framework includes comprehensive error handling:

- **Configuration Errors**: Invalid config files or missing parameters
- **Memory System Errors**: Connection failures or API errors
- **Attack/Defense Errors**: Execution failures with detailed error messages
- **Evaluation Errors**: Metric calculation failures with fallbacks

All errors are logged with timestamps and context information for debugging.

## Performance Considerations

- **Memory Usage**: Large test datasets may require significant RAM
- **Execution Time**: Comprehensive benchmarks can take hours to complete
- **API Limits**: External memory systems may have rate limits
- **Parallelization**: Consider distributing experiments across multiple machines

## Security Notes

- **API Keys**: Store securely and never commit to version control
- **Test Data**: Use synthetic data to avoid exposing sensitive information
- **Logging**: Be cautious with verbose logging of attack payloads
- **External Systems**: Verify security practices of integrated memory systems

## Contributing

When extending the framework:

1. Follow the established patterns for new attacks/defenses
2. Add comprehensive tests for new components
3. Update documentation and type hints
4. Ensure backward compatibility
5. Run full test suite before committing

## License

This research framework is provided for academic and research purposes. Please cite appropriately in publications.
