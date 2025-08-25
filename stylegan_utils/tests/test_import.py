def test_import_package():
    try:
        import stylegan_utils
    except ImportError as e:
        assert False, f"Failed to import stylegan_utils: {e}"

def test_submodules():
    import stylegan_utils
    assert hasattr(stylegan_utils, "dnnlib")
    assert hasattr(stylegan_utils, "legacy")
    assert hasattr(stylegan_utils, "torch_utils")

test_import_package()
test_submodules()
