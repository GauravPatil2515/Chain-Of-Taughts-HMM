# Contributing to CoT-HMM

Thank you for your interest in contributing to CoT-HMM! This document provides guidelines for contributing to this research project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/chain-of-thought-hmm.git
   cd chain-of-thought-hmm
   ```

3. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 📋 Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `experiment/description` - New experiments

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible

### Commit Messages

Use conventional commits:

```
feat: add neural tagger option
fix: handle empty tag sequences
docs: update installation instructions
exp: add StrategyQA evaluation
```

## 🧪 Running Tests

Before submitting:

```bash
# Run the demo pipeline to verify everything works
python demo_pipeline.py

# Quick sanity check
python -c "from src.hmm.scorer import HMMScorer; print('OK')"
```

## 📝 Areas for Contribution

### High Priority

- [ ] Neural-based semantic tagger (replace regex)
- [ ] Additional dataset support (StrategyQA, MATH)
- [ ] More LLM backends (Llama, Phi, DeepSeek)

### Medium Priority

- [ ] Visualization dashboard
- [ ] Streaming/real-time scoring API
- [ ] Docker containerization

### Documentation

- [ ] API reference docs
- [ ] Jupyter notebook tutorials
- [ ] Video walkthrough

## 🐛 Reporting Issues

When reporting bugs, include:

1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant error messages

## 📧 Questions?

Contact: <gauravppaiml123@gst.sies.edu.in>
