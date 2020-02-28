try:
    from custom._finalizer import finalizer
except ImportError:
    def finalizer(res):
        return res
