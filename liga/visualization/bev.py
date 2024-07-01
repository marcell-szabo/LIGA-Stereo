from typing import Optional, Any, Dict, Union, List
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

import torch

def tensor2ndarray(value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """If the type of value is torch.Tensor, convert the value to np.ndarray.

    Args:
        value (np.ndarray, torch.Tensor): value.

    Returns:
        Any: value.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return value

class BEVVisualizer:
    scale: int = 10

    def __init__(self, fig_cfg: Dict[str, Any]):
        fig = Figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)
        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        
        canvas = FigureCanvasAgg(fig)
        self.cavas, self.fig, self.ax_save =  canvas, fig, ax

    def set_bev_image(self,
                        bev_image: Optional[np.ndarray] = None,
                        bev_shape: Optional[int] = 800) -> None:
            """Set the bev image to draw.

            Args:
                bev_image (np.ndarray, optional): The bev image to draw.
                    Defaults to None.
                bev_shape (int): The bev image shape. Defaults to 900.
            """
            if bev_image is None:
                bev_image = np.zeros((bev_shape, bev_shape, 3), np.uint8)

            self._image = bev_image
            self.width, self.height = bev_image.shape[1], bev_image.shape[0]
            self._default_font_size = max(
                np.sqrt(self.height * self.width) // 90, 10)
            self.ax_save.cla()
            self.ax_save.axis(False)
            self.ax_save.imshow(bev_image, origin='lower')
            for i in range(0, self.width, 100):
                self.ax_save.axvline(i, color='white', linestyle='-', linewidth=1,alpha=0.1)
                self.ax_save.text(i, 0, str(np.abs(self.width/2 - i)), color='white', fontsize=8, ha='center', va='bottom')
            for j in range(0, self.height, 100):
                self.ax_save.axhline(j, color='white', linestyle='-', linewidth=1, alpha=0.3)
                self.ax_save.text(0, j, str(j/self.scale), color='white', fontsize=8, ha='left', va='center')

            # plot camera view range
            x1 = np.linspace(0, self.width / 2)
            x2 = np.linspace(self.width / 2, self.width)
            self.ax_save.plot(
                x1,
                self.width / 2 - x1,
                ls='--',
                color='grey',
                linewidth=1,
                alpha=0.5)
            self.ax_save.plot(
                x2,
                x2 - self.width / 2,
                ls='--',
                color='grey',
                linewidth=1,
                alpha=0.5)
            self.ax_save.plot(
                self.width / 2,
                0,
                marker='+',
                markersize=16,
            markeredgecolor='red')

    def draw_bev_bboxes(self,
                        bboxes_3d: Any,
                        path: str,
                        edge_colors: Union[str, tuple, List[str],
                                        List[tuple]] = 'o',
                        line_styles: Union[str, List[str]] = '-',
                        line_widths: Union[Union[int, float],
                                        List[Union[int, float]]] = 1,
                        face_colors: Union[str, tuple, List[str],
                                        List[tuple]] = 'none',
                        alpha: Union[int, float] = 1):
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`, shape=[M, 7]):
                3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            scale (dict): Value to scale the bev bboxes for better
                visualization. Defaults to 15.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'o'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Default to 'none'.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 1.
        """

        bev_bboxes = tensor2ndarray(bboxes_3d[:, [0, 2, 4, 5, 6]])
        bev_bboxes[:, -1] = -bev_bboxes[:, -1]
        # scale the bev bboxes for better visualization
        bev_bboxes[:, :4] *= self.scale
        ctr, w, h, theta = np.split(bev_bboxes, [2, 3, 4], axis=-1)
        cos_value, sin_value = np.cos(theta), np.sin(theta)
        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        poly = np.stack([pt1, pt2, pt3, pt4], axis=-2)
        # move the object along x-axis
        poly[:, :, 0] += self.width / 2
        poly = [p for p in poly]
        self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)
        self.fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches = 0)

    def _is_posion_valid(self, position: np.ndarray) -> bool:
        """Judge whether the position is in image.

        Args:
            position (np.ndarray): The position to judge which last dim must
                be two and the format is [x, y].

        Returns:
            bool: Whether the position is in image.
        """
        flag = (position[..., 0] < self.width).all() and \
               (position[..., 0] >= 0).all() and \
               (position[..., 1] < self.height).all() and \
               (position[..., 1] >= 0).all()
        return flag        

    def draw_polygons(
            self,
            polygons: Union[Union[np.ndarray, torch.Tensor],
                            List[Union[np.ndarray, torch.Tensor]]],
            edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
            face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
            alpha: Union[int, float] = 0.8,
        ) -> 'Visualizer':
            """Draw single or multiple bboxes.

            Args:
                polygons (Union[Union[np.ndarray, torch.Tensor],\
                    List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                    with the format of (x1,y1,x2,y2,...,xn,yn).
                edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                    colors of polygons. ``colors`` can have the same length with
                    lines or just single value. If ``colors`` is single value,
                    all the lines will have the same colors. Refer to
                    `matplotlib.colors` for full list of formats that are accepted.
                    Defaults to 'g.
                line_styles (Union[str, List[str]]): The linestyle
                    of lines. ``line_styles`` can have the same length with
                    texts or just single value. If ``line_styles`` is single
                    value, all the lines will have the same linestyle.
                    Reference to
                    https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                    for more details. Defaults to '-'.
                line_widths (Union[Union[int, float], List[Union[int, float]]]):
                    The linewidth of lines. ``line_widths`` can have
                    the same length with lines or just single value.
                    If ``line_widths`` is single value, all the lines will
                    have the same linewidth. Defaults to 2.
                face_colors (Union[str, tuple, List[str], List[tuple]]):
                    The face colors. Defaults to None.
                alpha (Union[int, float]): The transparency of polygons.
                    Defaults to 0.8.
            """
            # check_type('polygons', polygons, (list, np.ndarray, torch.Tensor))
            # edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
            # face_colors = color_val_matplotlib(face_colors)  # type: ignore

            if isinstance(polygons, (np.ndarray, torch.Tensor)):
                polygons = [polygons]
            if isinstance(polygons, list):
                for polygon in polygons:
                    assert polygon.shape[1] == 2, (
                        'The shape of each polygon in `polygons` should be (M, 2),'
                        f' but got {polygon.shape}')
            polygons = [tensor2ndarray(polygon) for polygon in polygons]
            for polygon in polygons:
                if not self._is_posion_valid(polygon):
                    print(
                        'Warning: The polygon is out of bounds,'
                        ' the drawn polygon may not be in the image', UserWarning)
            if isinstance(line_widths, (int, float)):
                line_widths = [line_widths] * len(polygons)
            line_widths = [
                min(max(linewidth, 1), self._default_font_size / 4)
                for linewidth in line_widths
            ]
            polygon_collection = PolyCollection(
                polygons,
                alpha=alpha,
                facecolor=face_colors,
                linestyles=line_styles,
                edgecolors=edge_colors,
                linewidths=line_widths)

            self.ax_save.add_collection(polygon_collection)