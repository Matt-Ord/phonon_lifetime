def test_import() -> None:
    try:
        import phonon_lifetime  # noqa: PLC0415
    except ImportError:
        phonon_lifetime = None

    assert phonon_lifetime is not None, "my_project module should not be None"
