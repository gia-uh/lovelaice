import pytest
from lovelaice.coding.tools.read import read


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello\nworld\n")
    out = await read.run(path=str(p))
    assert "hello" in out
    assert "world" in out


@pytest.mark.asyncio
async def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await read.run(path=str(tmp_path / "nope.txt"))


@pytest.mark.asyncio
async def test_read_directory_raises(tmp_path):
    with pytest.raises(IsADirectoryError):
        await read.run(path=str(tmp_path))


@pytest.mark.asyncio
async def test_read_truncates_long_file(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("line\n" * 5000)
    out = await read.run(path=str(p))
    assert "[truncated" in out
