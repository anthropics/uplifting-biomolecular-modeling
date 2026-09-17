try:
    import bg_hook
except Exception as _e:
    import sys; print('bg hook failed', repr(_e), file=sys.stderr)
