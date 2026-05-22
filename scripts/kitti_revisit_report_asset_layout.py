"""Layout geometry for the KITTI revisit README asset renderer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportAssetLayout:
    width: int
    margin: int = 36
    header_height: int = 140
    footer_height: int = 118
    overlay_aspect: float = 0.21

    @property
    def overlay_width(self) -> int:
        return self.width - self.margin * 2

    @property
    def overlay_height(self) -> int:
        return round(self.overlay_width * self.overlay_aspect)

    @property
    def height(self) -> int:
        return self.header_height + self.overlay_height + self.footer_height + self.margin

    @property
    def overlay_xy(self) -> tuple[int, int]:
        return self.margin, self.header_height

    @property
    def overlay_size(self) -> tuple[int, int]:
        return self.overlay_width, self.overlay_height

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def footer_y(self) -> int:
        return self.header_height + self.overlay_height + 24

    @property
    def settings_xy(self) -> tuple[int, int]:
        return self.margin, self.footer_y + 58

    def badge_xy(self, index: int) -> tuple[int, int]:
        offsets = (0, 270, 555, 745)
        return self.margin + offsets[index], self.footer_y
