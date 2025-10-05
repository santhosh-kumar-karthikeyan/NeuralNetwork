"""
Views for output formatting.
"""
from tabulate import tabulate

def print_table(rows, headers):
    """Print a table using tabulate in simple format."""
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="center"))