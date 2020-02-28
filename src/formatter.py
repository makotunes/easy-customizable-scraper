try:
    from custom._formatter import formatter
except ImportError:
    def formatter(sel):
        res = {}
        return res