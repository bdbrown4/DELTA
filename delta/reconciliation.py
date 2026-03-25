"""
Reconciliation Layer for DELTA.

After the dual parallel attention fires (nodes and edges updated independently),
the reconciliation layer ensures bidirectional information flow:
- Updated node features inform edge features
- Updated edge features inform node features

This is re-exported from attention.py where ReconciliationBridge is implemented
alongside the attention modules it serves.
"""

from delta.attention import ReconciliationBridge

ReconciliationLayer = ReconciliationBridge
