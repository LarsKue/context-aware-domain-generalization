from torchvision.datasets.utils import download_url, extract_archive
from pathlib import Path
import shutil


class DomainBedDownloadMixin:
    root: Path

    url: str
    filename: str
    dirname: str
    md5: str
    _remove_finished: bool = True

    def download(self):
        archive = self.root / self.filename
        target = self.root / self.dirname

        if target.is_dir():
            print(f"Found existing {self.__class__.__name__} in {self.root}, skipping download.")
            return

        print(f"Downloading {self.__class__.__name__} dataset from {self.url} to {self.root}...")

        download_url(
            url=self.url,
            root=str(self.root),
            filename=self.filename,
            md5=self.md5,
        )

        assert archive.is_file(), f"Failed to download {self.__class__.__name__} to archive file."

        print(f"Extracting {archive} to {self.root}...")
        extract_archive(
            from_path=str(archive),
            to_path=str(target),
            remove_finished=self._remove_finished,
        )

        assert target.is_dir(), f"Failed to extract {self.__class__.__name__} archive to directory."

        # remove intermediate folder by moving all files up one level
        intermediate_folders = list(target.glob("*"))
        assert len(intermediate_folders) == 1, f"Expected exactly one intermediate folder, " \
                                               f"found {len(intermediate_folders)}"

        # move all files up one level
        intermediate_folder = intermediate_folders[0]
        for folder in intermediate_folder.iterdir():
            if folder.is_dir():
                shutil.move(str(folder), str(target / folder.name))

        # remove intermediate folder
        shutil.rmtree(str(intermediate_folder))
