"""
lob_simulator — High-performance Limit Order Book matching engine.

Modules
-------
core.types   : Price, Qty, Side, OrderType, enums
core.order   : Order node (data + linked-list pointers)
core.level   : LimitLevel (doubly-linked list at one price)
core.book    : Book (CLOB matching engine)
metrics.execution : Slippage, fill probability, market impact
"""
