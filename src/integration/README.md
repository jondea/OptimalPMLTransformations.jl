# Integration

## Adaptive integration

Whereas a typical binary tree will split elements in half, we could do a "directed" split by taking one step of the Newton method for pole finding.
This should find poles faster than a binary search, and by storing the field at that point, we wouldn't be throwing away any field information.
