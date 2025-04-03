#!/bin/bash

# Funes Installation Script
# This script installs Funes and all its dependencies without Docker

set -e  # Exit on error

# Text formatting
BOLD="\e[1m"
GREEN="\e[32m"
BLUE="\e[34m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

echo -e "${BOLD}${BLUE}=====================================================${RESET}"
echo -e "${BOLD}${BLUE}           Funes Installation Script                 ${RESET}"
echo -e "${BOLD}${BLUE}=====================================================${RESET}"

# Check if script is run as root
if [ "$(id -u)" -eq 0 ]; then
    echo -e "${BOLD}${RED}This script should not be run as root or with sudo.${RESET}"
    echo -e "Please run as a regular user with sudo privileges."
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to print section header
print_section() {
    echo -e "\n${BOLD}${BLUE}[+] $1${RESET}"
}

# Function to print success message
print_success() {
    echo -e "${BOLD}${GREEN}✓ $1${RESET}"
}

# Function to print warning message
print_warning() {
    echo -e "${BOLD}${YELLOW}! $1${RESET}"
}

# Function to print error message
print_error() {
    echo -e "${BOLD}${RED}✗ $1${RESET}"
}

# Check system requirements
print_section "Checking system requirements"

# Check OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME=$NAME
    OS_VERSION=$VERSION_ID
    echo "Detected OS: $OS_NAME $OS_VERSION"
else
    print_error "Unable to determine operating system."
    exit 1
fi

# Check for supported OS
case $OS_NAME in
    *"Ubuntu"*|*"Debian"*)
        print_success "Running on supported OS: $OS_NAME"
        ;;
    *)
        print_warning "Running on untested OS: $OS_NAME. Installation may not work correctly."
        sleep 2
        ;;
esac

# Check for required commands
for cmd in curl wget git python3 pip3; do
    if command_exists $cmd; then
        echo "$cmd is installed"
    else
        print_error "$cmd is not installed. Please install it before continuing."
        exit 1
    fi
done

# Install system dependencies
print_section "Installing system dependencies"

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libxkbcommon0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    xvfb \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all

print_success "System dependencies installed"

# Install Ollama
print_section "Installing Ollama"

if command_exists ollama; then
    print_warning "Ollama is already installed. Skipping installation."
else
    curl -fsSL https://ollama.ai/install.sh | sh
    print_success "Ollama installed successfully"
fi

# Check if PostgreSQL is running
print_section "Checking PostgreSQL service"

if systemctl is-active --quiet postgresql; then
    print_success "PostgreSQL service is running"
else
    print_warning "PostgreSQL service is not running. Attempting to start..."
    sudo systemctl start postgresql
    if systemctl is-active --quiet postgresql; then
        print_success "PostgreSQL service started successfully"
    else
        print_error "Failed to start PostgreSQL service. Please check PostgreSQL installation."
        exit 1
    fi
fi

# Install pgvector
print_section "Installing pgvector extension"

PGVECTOR_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
pushd "$PGVECTOR_DIR"
make
sudo make install
popd
rm -rf "$PGVECTOR_DIR"
print_success "pgvector extension installed"

# Create PostgreSQL user and database
print_section "Setting up PostgreSQL database for Funes"

# Create user if it doesn't exist
sudo -u postgres psql -c "DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'llm') THEN
        CREATE USER llm WITH PASSWORD 'llm';
    END IF;
END
\$\$;" || true

# Create database if it doesn't exist
sudo -u postgres psql -c "SELECT 'CREATE DATABASE funes' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'funes')\gexec" || true

# Grant proper permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE funes TO llm;" || true
sudo -u postgres psql -d funes -c "GRANT ALL PRIVILEGES ON SCHEMA public TO llm;" || true
sudo -u postgres psql -d funes -c "ALTER USER llm WITH LOGIN CREATEDB CREATEROLE;" || true

# Create the vector extension
sudo -u postgres psql -d funes -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Set up database tables for vector-based tool selection
print_section "Setting up vector-based tool selection"

# Create the tools_embeddings table
sudo -u postgres psql -d funes -c "
CREATE TABLE IF NOT EXISTS tools_embeddings (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(100) UNIQUE,
    description TEXT,
    embedding vector(384),
    updated_at TIMESTAMP DEFAULT NOW()
);" || true

# Create index on tool embeddings for faster similarity search
sudo -u postgres psql -d funes -c "
CREATE INDEX IF NOT EXISTS idx_tools_embedding 
ON tools_embeddings 
USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 100);" || true

print_success "PostgreSQL database setup completed with proper permissions"
print_success "Vector-based tool selection tables created"

# Set up Python environment
print_section "Setting up Python environment"

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"


# Create virtual environment
sudo apt install python3.11-venv
python3 -m venv funes-env
source funes-env/bin/activate

# Generate requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    cat > requirements.txt << EOL
sentence-transformers>=2.2.0
psycopg2-binary>=2.9.5
PySide2>=5.15.0
ollama>=0.1.0
docling>=0.1.0
numpy>=1.24.0
python-dotenv>=1.0.0
EOL
    print_success "Generated requirements.txt"
else
    print_warning "Using existing requirements.txt"
fi

pip install --upgrade pip
pip install -r requirements.txt

print_success "Python environment setup completed"

# Create .env file
print_section "Creating environment configuration"

if [ ! -f .env ]; then
    cat > .env << EOL
DATABASE_URL=postgresql://llm:llm@localhost:5432/funes
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=llm
POSTGRES_PASSWORD=llm
POSTGRES_DB=funes
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
EOL
    print_success "Created .env file with default settings"
else
    print_warning "Using existing .env file"
fi

# Create a launcher script
print_section "Creating launcher script"

cat > run_funes.sh << EOL
#!/bin/bash

# Activate virtual environment
source funes-env/bin/activate

# Start Xvfb for GUI on headless systems (if needed)
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=\$!

# Run Funes
python $(pwd)/funes.py

# Clean up
kill \$XVFB_PID
EOL

chmod +x run_funes.sh
print_success "Created launcher script: run_funes.sh"

# Start Ollama service
print_section "Starting Ollama service"

# Check if Ollama service exists and is active
if systemctl list-unit-files | grep -q ollama.service; then
    sudo systemctl restart ollama
    print_success "Ollama service restarted"
else
    echo "Starting Ollama manually..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 2
    print_success "Ollama started manually (PID: $OLLAMA_PID)"
fi

# Download required models
print_section "Downloading required LLM models"

ollama pull smallthinker
print_success "Downloaded required models"

# Final instructions
print_section "Installation Complete!"

echo -e "${BOLD}To run Funes:${RESET}"
echo -e "  1. Make sure the PostgreSQL service is running:"
echo -e "     ${YELLOW}sudo systemctl start postgresql${RESET}"
echo -e "  2. Make sure Ollama is running:"
echo -e "     ${YELLOW}ollama serve${RESET}"
echo -e "  3. Run Funes using the launcher script:"
echo -e "     ${YELLOW}./run_funes.sh${RESET}"

echo -e "\n${BOLD}Features:${RESET}"
echo -e "  • Dual Memory System (short-term and long-term)"
echo -e "  • Retrieval-Augmented Generation (RAG)"
echo -e "  • Vector-based Tool Selection"
echo -e "  • Automatic tool embedding initialization"
echo -e "  • Smart memory management (tool interactions not stored in long-term memory)"

echo -e "\n${BOLD}${GREEN}Thank you for installing Funes!${RESET}"
echo -e "${BOLD}${BLUE}=====================================================${RESET}"