"""Streaming speaker-diarization support for AST v3 sessions.

Submodules are intentionally not re-exported here: ``turns`` depends on the
streaming event types, while ``session`` owns the gRPC client. Eager imports at
package initialization would therefore create a circular dependency.
"""
