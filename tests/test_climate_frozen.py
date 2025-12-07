import pytest


def test_climate_folder_is_frozen():
    with pytest.raises(ImportError):
        import climate
