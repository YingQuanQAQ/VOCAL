def within_cuda_device(f):
    def _wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return _wrapper
