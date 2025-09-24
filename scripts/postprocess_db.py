#!/usr/bin/env python3
"""
Database Post-processing Script for Trading Signal Labeler

This script cleans up and standardizes the labeled data in the database:
1. Sets all extended labels to None for non-signals (Signal=False)
2. Standardizes Direction field to only SHORT/LONG
3. Extracts base coin from trading pairs (e.g., BTCUSDT -> BTC)
4. Reports incomplete signals missing Direction or Pair
"""

import sqlite3
import re
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
    ]
)
logger = logging.getLogger(__name__)

# Database file
DB_FILE = "total.db"

# Direction mapping
DIRECTION_MAPPING = {
    # Standard forms
    'long': 'LONG',
    'short': 'SHORT',
    'buy': 'LONG',
    'sell': 'SHORT',
    # Numeric forms
    '1': 'LONG',
    '0': 'SHORT',
    # Other variants
    'l': 'LONG',
    's': 'SHORT',
    'b': 'LONG',
    'bullish': 'LONG',
    'bearish': 'SHORT',
    'up': 'LONG',
    'down': 'SHORT',
}

# Common trading pair suffixes to remove
PAIR_SUFFIXES = [
    'USDT', 'USDC', 'USD', 'BUSD', 'DAI', 'TUSD', 'USDD',
    'BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL', 'AVAX',
    'MATIC', 'ATOM', 'NEAR', 'FTM', 'ALGO', 'XRP', 'LTC',
    'DOGE', 'SHIB', 'TRX', 'EOS', 'XLM', 'VET', 'FIL',
    'THETA', 'AXS', 'SAND', 'MANA', 'ENJ', 'CHZ', 'BAT'
]

def setup_database(db_file):
    """Setup database connection."""
    logger.info("🗄️  Connecting to database...")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        logger.info(f"✅ Database connection established: [bold green]{db_file}[/bold green]")
        return conn, cursor
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise

def standardize_direction(direction_str):
    """
    Standardize direction to LONG or SHORT.
    
    Args:
        direction_str: Original direction string
        
    Returns:
        str: Standardized direction (LONG/SHORT) or None if cannot be determined
    """
    if not direction_str:
        return None
    
    # Clean and normalize the input
    cleaned = str(direction_str).strip().lower()
    
    # Check direct mapping
    if cleaned in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[cleaned]
    
    # Check if it contains keywords
    for key, value in DIRECTION_MAPPING.items():
        if key in cleaned:
            return value
    
    # If we can't determine, return None
    return None

def extract_base_pair(pair_str):
    """
    Extract base coin from trading pair.
    
    Args:
        pair_str: Original pair string (e.g., 'BTCUSDT', 'BTC/USDT', 'BTC-USD')
        
    Returns:
        str: Base coin (e.g., 'BTC') or original string if cannot be determined
    """
    if not pair_str:
        return None
    
    # Clean the input
    cleaned = str(pair_str).strip().upper()
    
    # Remove common separators
    cleaned = re.sub(r'[/\-_]', '', cleaned)
    
    # Try to extract base coin by removing known suffixes
    for suffix in sorted(PAIR_SUFFIXES, key=len, reverse=True):  # Longest first
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            base_coin = cleaned[:-len(suffix)]
            # Validate that the remaining part looks like a coin symbol
            if len(base_coin) >= 2 and base_coin.isalpha():
                return base_coin
    
    # If no suffix matches, check if it's already a single coin
    if cleaned.isalpha() and 2 <= len(cleaned) <= 6:
        return cleaned
    
    # Return original if we can't process it
    return pair_str

def get_database_stats(cursor):
    """Get current database statistics."""
    stats = {}
    
    # Total labeled messages
    cursor.execute("SELECT COUNT(*) FROM labeled")
    stats['total'] = cursor.fetchone()[0]
    
    # Signals vs non-signals
    cursor.execute("SELECT COUNT(*) FROM labeled WHERE is_signal = 1")
    stats['signals'] = cursor.fetchone()[0]
    stats['non_signals'] = stats['total'] - stats['signals']
    
    # Signals with missing direction
    cursor.execute("SELECT COUNT(*) FROM labeled WHERE is_signal = 1 AND (direction IS NULL OR direction = '')")
    stats['missing_direction'] = cursor.fetchone()[0]
    
    # Signals with missing pair
    cursor.execute("SELECT COUNT(*) FROM labeled WHERE is_signal = 1 AND (pair IS NULL OR pair = '')")
    stats['missing_pair'] = cursor.fetchone()[0]
    
    return stats

def process_non_signals(conn, cursor):
    """Process non-signal entries - set all extended labels to None."""
    logger.info("🔄 Processing non-signal entries...")
    
    cursor.execute("""
        UPDATE labeled 
        SET pair = NULL, direction = NULL, entry = NULL, targets = NULL, stop_loss = NULL, leverage = NULL
        WHERE is_signal = 0 OR is_signal IS NULL
    """)
    
    updated_count = cursor.rowcount
    conn.commit()
    
    logger.info(f"✅ Updated {updated_count} non-signal entries")
    return updated_count

def process_signal_directions(conn, cursor):
    """Process and standardize direction fields for signals."""
    logger.info("🔄 Processing signal directions...")
    
    # Get all signals with directions
    cursor.execute("SELECT id, direction FROM labeled WHERE is_signal = 1 AND direction IS NOT NULL AND direction != ''")
    rows = cursor.fetchall()
    
    updated_count = 0
    standardized_count = 0
    
    for row_id, direction in rows:
        original_direction = direction
        standardized = standardize_direction(direction)
        
        if standardized and standardized != original_direction:
            cursor.execute("UPDATE labeled SET direction = ? WHERE id = ?", (standardized, row_id))
            updated_count += 1
            logger.debug(f"📝 Updated direction: '{original_direction}' -> '{standardized}' (ID: {row_id})")
        elif standardized:
            standardized_count += 1
    
    conn.commit()
    logger.info(f"✅ Standardized {updated_count} direction entries ({standardized_count} already correct)")
    return updated_count

def process_signal_pairs(conn, cursor):
    """Process and standardize pair fields for signals."""
    logger.info("🔄 Processing signal pairs...")
    
    # Get all signals with pairs
    cursor.execute("SELECT id, pair FROM labeled WHERE is_signal = 1 AND pair IS NOT NULL AND pair != ''")
    rows = cursor.fetchall()
    
    updated_count = 0
    standardized_count = 0
    
    for row_id, pair in rows:
        original_pair = pair
        standardized = extract_base_pair(pair)
        
        if standardized and standardized != original_pair:
            cursor.execute("UPDATE labeled SET pair = ? WHERE id = ?", (standardized, row_id))
            updated_count += 1
            logger.debug(f"📝 Updated pair: '{original_pair}' -> '{standardized}' (ID: {row_id})")
        elif standardized:
            standardized_count += 1
    
    conn.commit()
    logger.info(f"✅ Standardized {updated_count} pair entries ({standardized_count} already correct)")
    return updated_count

def report_incomplete_signals(cursor):
    """Report signals that are missing required fields."""
    logger.info("🔍 Checking for incomplete signals...")
    
    # Signals missing direction
    cursor.execute("""
        SELECT id, message_id, message, pair 
        FROM labeled 
        WHERE is_signal = 1 AND (direction IS NULL OR direction = '')
    """)
    missing_direction = cursor.fetchall()
    
    # Signals missing pair
    cursor.execute("""
        SELECT id, message_id, message, direction 
        FROM labeled 
        WHERE is_signal = 1 AND (pair IS NULL OR pair = '')
    """)
    missing_pair = cursor.fetchall()
    
    # Report missing directions
    if missing_direction:
        console.print(f"\n⚠️  [bold yellow]Found {len(missing_direction)} signals missing DIRECTION:[/bold yellow]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Message ID", style="yellow", no_wrap=True)
        table.add_column("Pair", style="green", no_wrap=True)
        table.add_column("Message Preview", style="white")
        
        for row_id, msg_id, message, pair in missing_direction[:10]:  # Show first 10
            message_preview = (message[:50] + "...") if len(message) > 50 else message
            table.add_row(str(row_id), str(msg_id), str(pair) if pair else "None", message_preview)
        
        console.print(table)
        
        if len(missing_direction) > 10:
            console.print(f"... and {len(missing_direction) - 10} more entries")
    else:
        logger.info("✅ All signals have valid directions")
    
    # Report missing pairs
    if missing_pair:
        console.print(f"\n⚠️  [bold yellow]Found {len(missing_pair)} signals missing PAIR:[/bold yellow]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Message ID", style="yellow", no_wrap=True)
        table.add_column("Direction", style="green", no_wrap=True)
        table.add_column("Message Preview", style="white")
        
        for row_id, msg_id, message, direction in missing_pair[:10]:  # Show first 10
            message_preview = (message[:50] + "...") if len(message) > 50 else message
            table.add_row(str(row_id), str(msg_id), str(direction) if direction else "None", message_preview)
        
        console.print(table)
        
        if len(missing_pair) > 10:
            console.print(f"... and {len(missing_pair) - 10} more entries")
    else:
        logger.info("✅ All signals have valid pairs")
    
    return len(missing_direction), len(missing_pair)

def remove_incomplete_signals(conn, cursor):
    """Remove signals that are missing required fields after user confirmation."""
    logger.info("🗑️  Checking for incomplete signals to remove...")
    
    # Get count of incomplete signals
    cursor.execute("""
        SELECT COUNT(*) FROM labeled 
        WHERE is_signal = 1 AND (
            (direction IS NULL OR direction = '') OR 
            (pair IS NULL OR pair = '')
        )
    """)
    incomplete_count = cursor.fetchone()[0]
    
    if incomplete_count == 0:
        logger.info("✅ No incomplete signals found to remove")
        return 0
    
    # Ask user for confirmation
    console.print(f"\n⚠️  [bold yellow]Found {incomplete_count} incomplete signals[/bold yellow]")
    console.print("These signals are missing either DIRECTION or PAIR information.")
    
    if Confirm.ask("Do you want to remove these incomplete signals?", default=False):
        # Remove incomplete signals
        cursor.execute("""
            DELETE FROM labeled 
            WHERE is_signal = 1 AND (
                (direction IS NULL OR direction = '') OR 
                (pair IS NULL OR pair = '')
            )
        """)
        
        removed_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"🗑️  Removed {removed_count} incomplete signals")
        console.print(f"✅ [bold green]Successfully removed {removed_count} incomplete signals[/bold green]")
        return removed_count
    else:
        logger.info("⏭️  Skipping removal of incomplete signals")
        return 0

def display_processing_summary(before_stats, after_stats, updates):
    """Display a summary of the processing results."""
    console.print("\n🎯 [bold green]Post-processing Complete![/bold green]")
    
    # Create summary table
    table = Table(title="📊 Post-processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Before", style="yellow", justify="right")
    table.add_column("After", style="green", justify="right")
    table.add_column("Changes", style="blue", justify="right")
    
    table.add_row("Total Messages", str(before_stats['total']), str(after_stats['total']), 
                  str(after_stats['total'] - before_stats['total']))
    table.add_row("Signals", str(before_stats['signals']), str(after_stats['signals']), 
                  str(after_stats['signals'] - before_stats['signals']))
    table.add_row("Non-signals", str(before_stats['non_signals']), str(after_stats['non_signals']), 
                  str(after_stats['non_signals'] - before_stats['non_signals']))
    table.add_row("Missing Direction", str(before_stats['missing_direction']), str(after_stats['missing_direction']), 
                  str(after_stats['missing_direction'] - before_stats['missing_direction']))
    table.add_row("Missing Pair", str(before_stats['missing_pair']), str(after_stats['missing_pair']), 
                  str(after_stats['missing_pair'] - before_stats['missing_pair']))
    
    console.print(table)
    
    # Update summary
    updates_table = Table(title="🔧 Updates Applied", show_header=True, header_style="bold cyan")
    updates_table.add_column("Operation", style="cyan")
    updates_table.add_column("Records Updated", style="green", justify="right")
    
    updates_table.add_row("Non-signal cleanup", str(updates['non_signals']))
    updates_table.add_row("Direction standardization", str(updates['directions']))
    updates_table.add_row("Pair standardization", str(updates['pairs']))
    if 'removed' in updates and updates['removed'] > 0:
        updates_table.add_row("Incomplete signals removed", str(updates['removed']))
    
    console.print(updates_table)

def main():
    """Main function to run the post-processing."""
    
    # Display welcome banner
    console.print(Panel.fit("🧹 [bold blue]Database Post-processor[/bold blue] 🧹", border_style="blue"))
    console.print("Cleaning and standardizing labeled trading signal data...")
    
    # Setup database
    conn, cursor = setup_database(DB_FILE)
    
    # Get initial statistics
    before_stats = get_database_stats(cursor)
    logger.info(f"📊 Initial stats: {before_stats['total']} total messages, {before_stats['signals']} signals")
    
    # Track updates
    updates = {
        'non_signals': 0,
        'directions': 0,
        'pairs': 0,
        'removed': 0
    }
    
    try:
        # Process non-signals
        updates['non_signals'] = process_non_signals(conn, cursor)
        
        # Process signal directions
        updates['directions'] = process_signal_directions(conn, cursor)
        
        # Process signal pairs
        updates['pairs'] = process_signal_pairs(conn, cursor)
        
        # Report incomplete signals
        missing_dir, missing_pair = report_incomplete_signals(cursor)
        
        # Ask if user wants to remove incomplete signals
        if missing_dir > 0 or missing_pair > 0:
            updates['removed'] = remove_incomplete_signals(conn, cursor)
        
        # Get final statistics (after potential removal)
        after_stats = get_database_stats(cursor)
        
        # Display summary
        display_processing_summary(before_stats, after_stats, updates)
        
        # Check if there are still incomplete signals after potential removal
        final_stats = get_database_stats(cursor)
        if final_stats['missing_direction'] > 0 or final_stats['missing_pair'] > 0:
            console.print("\n⚠️  [bold red]Warning:[/bold red] Found incomplete signals that may need manual review")
            
        logger.info("✅ Post-processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during post-processing: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("🔒 Database connection closed")

if __name__ == "__main__":
    main()
