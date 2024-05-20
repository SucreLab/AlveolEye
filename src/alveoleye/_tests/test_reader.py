from alveoleye import napari_get_reader


def test_get_reader_pass():
    '''
    Test if the reader passes on files that it doesn't support
    '''
    reader = napari_get_reader("fake.file")
    assert reader is None
