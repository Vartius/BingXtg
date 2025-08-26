#!/bin/bash
# Cleanup script to remove old Django project directories and organize the project structure

echo "🧹 Cleaning up old Django project structure..."

# Remove old Django project directories (backup first)
if [ -d "trading_bot_project" ]; then
    echo "📦 Backing up trading_bot_project to bkp/"
    mkdir -p bkp/
    cp -r trading_bot_project bkp/trading_bot_project_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old trading_bot_project directory"
    rm -rf trading_bot_project
fi

if [ -d "webui" ]; then
    echo "📦 Backing up webui to bkp/"
    mkdir -p bkp/
    cp -r webui bkp/webui_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old webui directory"
    rm -rf webui
fi

if [ -d "config" ]; then
    echo "📦 Backing up config to bkp/"
    mkdir -p bkp/
    cp -r config bkp/config_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old config directory"
    rm -rf config
fi

# Remove old directories that are now reorganized
if [ -d "bot" ]; then
    echo "📦 Backing up bot directory to bkp/"
    mkdir -p bkp/
    cp -r bot bkp/bot_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old bot directory (now in apps/trading_bot)"
    rm -rf bot
fi

if [ -d "webapp" ]; then
    echo "📦 Backing up webapp directory to bkp/"
    mkdir -p bkp/
    cp -r webapp bkp/webapp_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old webapp directory (now in apps/ai_assistant)"
    rm -rf webapp
fi

if [ -d "src" ]; then
    echo "📦 Backing up src directory to bkp/"
    mkdir -p bkp/
    cp -r src bkp/src_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old src directory (now in utils and apps)"
    rm -rf src
fi

if [ -d "core" ]; then
    echo "📦 Backing up core directory to bkp/"
    mkdir -p bkp/
    cp -r core bkp/core_backup_$(date +%Y%m%d_%H%M%S)
    echo "🗑️  Removing old core directory (now in utils)"
    rm -rf core
fi

# Clean up Python cache directories
echo "🧹 Cleaning up Python cache files..."
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true

echo "✅ Cleanup completed!"
echo ""
echo "📁 Current project structure:"
echo "├── apps/                    # Django applications"
echo "│   ├── ai_assistant/        # AI classification and training"
echo "│   ├── telegram_client/     # Telegram integration"
echo "│   └── trading_bot/         # Main trading bot functionality"
echo "├── bingxtg_project/         # Main Django project"
echo "├── utils/                   # Shared utilities"
echo "├── static/                  # Static files"
echo "├── templates/               # Global templates"
echo "├── data/                    # Configuration files"
echo "└── manage.py                # Django management script"
echo ""
echo "🚀 To run the project:"
echo "   python manage.py runserver"
echo ""
echo "📚 See PROJECT_STRUCTURE.md for detailed documentation"
