import os
import sys
import json
import pandas as pd
from io import BytesIO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from winsight_predictions.nfl.players.data_object import DataObject

class MockS3Body:
    def __init__(self, data: str):
        self._data = data.encode("utf-8")
    def read(self):
        return self._data

class MockS3Client:
    def __init__(self):
        # Preload some mock objects keyed by S3 key
        self.objects = {
            "nfl/html_tables/sample.csv": "col1,col2\n1,2\n3,4\n",
            "nfl/html_tables/other.json": json.dumps({"hello": "world", "value": 42}),
            "nfl/html_tables/raw.txt": "Just some text here",
        }
    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = [k for k in self.objects.keys() if k.startswith(Prefix)]
        contents = [{"Key": k} for k in keys]
        return {"Contents": contents}
    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"Body": MockS3Body(self.objects[Key])}


def test_s3_listing_and_fetching():
    mock_client = MockS3Client()
    data_obj = DataObject(league="nfl", storage_mode="s3", s3_bucket="dummy-bucket", s3_client=mock_client)

    # Validate listing returns filenames (stripped of prefix) sorted
    files = data_obj.list_files(data_obj.local_data_dir)
    assert files == ["other.json", "raw.txt", "sample.csv"] or files == ["sample.csv", "other.json", "raw.txt"]  # order may vary; just ensure presence
    assert set(files) == {"sample.csv", "other.json", "raw.txt"}

    # CSV fetch
    df = data_obj.get_file("nfl/html_tables/sample.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

    # JSON fetch
    obj = data_obj.get_file("nfl/html_tables/other.json")
    assert obj["hello"] == "world" and obj["value"] == 42

    # TEXT fetch
    txt = data_obj.get_file("nfl/html_tables/raw.txt")
    assert txt.startswith("Just some text")


def test_get_file_type_override():
    mock_client = MockS3Client()
    data_obj = DataObject(league="nfl", storage_mode="s3", s3_bucket="dummy-bucket", s3_client=mock_client)
    # Force treating .csv as text
    raw = data_obj.get_file("nfl/html_tables/sample.csv", file_type="text")
    assert isinstance(raw, str)
    assert "col1,col2" in raw
