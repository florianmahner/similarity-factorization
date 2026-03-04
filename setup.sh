#!/bin/bash
set -e

echo "🚀 Setting up pysrf development environment..."

if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv not found. Please install pyenv first:"
    echo "   curl https://pyenv.run | bash"
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "❌ poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

PYTHON_VERSION="3.12.4"
echo "📦 Checking Python ${PYTHON_VERSION}..."
if ! pyenv versions | grep -q "${PYTHON_VERSION}"; then
    echo "Installing Python ${PYTHON_VERSION}..."
    pyenv install ${PYTHON_VERSION}
fi

echo "🔧 Setting local Python version..."
pyenv local ${PYTHON_VERSION}

echo "📚 Installing dependencies with Poetry..."
poetry install

echo "⚙️  Compiling Cython extensions..."
make compile || echo "⚠️  Cython compilation failed, will use Python fallback"

echo "🧪 Running tests..."
poetry run pytest tests/ -v

echo "✅ Setup complete! Activate environment with: poetry shell"

