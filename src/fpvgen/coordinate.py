from dataclasses import dataclass
from enum import Enum
import numpy as np
import h5py


class CoordType(Enum):
    """Enumeration of possible coordinate types for flamelet table generation"""

    NONSPECIFIC = 0
    LINEAR = 1
    POWER_LAW = 2
    TWO_LINEAR = 3
    LINEAR_THEN_STRETCHED = 4


class Coordinate:
    """Represents a coordinate in a flamelet table

    Attributes:
        name (str): The name of the coordinate.
        type (CoordType): The type of the coordinate.
        grid (np.ndarray): The grid values of the coordinate.
    """

    name: str
    type: CoordType
    grid: np.ndarray

    def __init__(self, name: str):
        """Initialize the Coordinate class.

        Args:
            name (str): The name of the coordinate.
            type (CoordType): The type of the coordinate.
            grid (np.ndarray): The grid values of the coordinate.
        """
        raise NotImplementedError("This is an abstract class")

    @property
    def lower_bound(self):
        """Get the lower bound of the grid.

        Returns:
            float: The first value in the grid.
        """
        return self.grid[0]

    @property
    def upper_bound(self):
        """Get the upper bound of the grid.

        Returns:
            float: The last value in the grid.
        """
        return self.grid[-1]

    @property
    def size(self):
        """Get the size of the grid.

        Returns:
            int: The number of elements in the grid.
        """
        return self.grid.size

    def write_hdf5(self, group: h5py.Group, dataset_name: str):
        """Write the coordinate data to an HDF5 group.

        Args:
            group (h5py.Group): The HDF5 group to write to.
            dataset_name (str): The name of the dataset.
        """
        group.create_dataset(dataset_name, data=self.grid)
        group.attrs["Name"] = self.name
        group.attrs["Type"] = int(self.type)
        group.attrs["Lower bound"] = self.lower_bound
        group.attrs["Upper bound"] = self.upper_bound
        group.attrs["Size"] = self.size


class CoordinateNonSpecific(Coordinate):
    """Represents a non-specific coordinate in a flamelet table

    Attributes:
        type (CoordType): The type of the coordinate (NONSPECIFIC).
        grid (np.ndarray): The grid values of the coordinate.
    """

    def __init__(self, name: str, grid: np.ndarray):
        """Initialize the CoordinateNonSpecific class.

        Args:
            grid (np.ndarray): The grid values of the coordinate.
        """
        self.name = name
        self.type = CoordType.NONSPECIFIC
        self.grid = grid


class CoordinateLinear(Coordinate):
    """Represents a linear coordinate in a flamelet table

    Attributes:
        type (CoordType): The type of the coordinate (LINEAR).
        grid (np.ndarray): The grid values of the coordinate.
    """

    def __init__(self, name: str, lower_bound: float, upper_bound: float, size: int):
        """Initialize the CoordinateLinear class.

        Args:
            lower_bound (float): The lower bound of the grid.
            upper_bound (float): The upper bound of the grid.
            size (int): The number of elements in the grid.
        """
        self.name = name
        self.type = CoordType.LINEAR
        self.grid = np.linspace(lower_bound, upper_bound, size)


class CoordinatePowerLaw(Coordinate):
    """Represents a power-law coordinate in a flamelet table

    Attributes:
        type (CoordType): The type of the coordinate (POWER_LAW).
        grid (np.ndarray): The grid values of the coordinate.
        growth_rate (float): The growth rate of the power-law.
    """

    def __init__(
        self, name: str, lower_bound: float, upper_bound: float, size: int, growth_rate: float
    ):
        """Initialize the CoordinatePowerLaw class.

        Args:
            lower_bound (float): The lower bound of the grid.
            upper_bound (float): The upper bound of the grid.
            size (int): The number of elements in the grid.
            growth_rate (float): The growth rate of the power-law.
        """
        self.name = name
        self.type = CoordType.POWER_LAW
        self.growth_rate = growth_rate
        self.grid = (
            np.linspace(
                lower_bound ** (1 / self.growth_rate), upper_bound ** (1 / self.growth_rate), size
            )
            ** self.growth_rate
        )

    def write_hdf5(self, group: h5py.Group, dataset_name: str):
        """Write the coordinate data to an HDF5 group, including the growth rate.

        Args:
            group (h5py.Group): The HDF5 group to write to.
            dataset_name (str): The name of the dataset.
        """
        super().write_hdf5(group, dataset_name)
        group.attrs["Growth rate"] = self.growth_rate


class CoordinateTwoLinear(Coordinate):
    """Represents a coordinate that is linear in two segments in a flamelet table

    Attributes:
        type (CoordType): The type of the coordinate (TWO_LINEAR).
        grid (np.ndarray): The grid values of the coordinate.
        middle (float): The middle value where the two linear segments meet.
        size1 (int): The number of elements in the first segment.
        size2 (int): The number of elements in the second segment.
    """

    def __init__(
        self,
        name: str,
        lower_bound: float,
        middle: float,
        upper_bound: float,
        size1: int,
        size2: int,
    ):
        """Initialize the CoordinateTwoLinear class.

        Args:
            lower_bound (float): The lower bound of the grid.
            middle (float): The middle value where the two linear segments meet.
            upper_bound (float): The upper bound of the grid.
            size1 (int): The number of elements in the first segment.
            size2 (int): The number of elements in the second segment.
        """
        self.name = name
        self.type = CoordType.TWO_LINEAR
        self.middle = middle
        self.size1 = size1
        self.size2 = size2
        self.grid = np.concatenate(
            (np.linspace(lower_bound, middle, size1), np.linspace(middle, upper_bound, size2))
        )

    def write_hdf5(self, group: h5py.Group, dataset_name: str):
        """Write the coordinate data to an HDF5 group, including the middle value and segment sizes.

        Args:
            group (h5py.Group): The HDF5 group to write to.
            dataset_name (str): The name of the dataset.
        """
        super().write_hdf5(group, dataset_name)
        group.attrs["i_cut"] = self.size1
        group.attrs["z_cut"] = self.middle


class CoordinateLinearThenStretched(Coordinate):
    """Represents a coordinate that is linear then stretched in a flamelet table

    Attributes:
        type (CoordType): The type of the coordinate (LINEAR_THEN_STRETCHED).
        grid (np.ndarray): The grid values of the coordinate.
        middle (float): The middle value where the linear segment ends and the stretched segment begins.
        size1 (int): The number of elements in the linear segment.
        size2 (int): The number of elements in the stretched segment.
    """

    def __init__(
        self,
        name: str,
        lower_bound: float,
        middle: float,
        upper_bound: float,
        size1: int,
        size2: int,
    ):
        """Initialize the CoordinateLinearThenStretched class.

        Args:
            lower_bound (float): The lower bound of the grid.
            middle (float): The middle value where the linear segment ends and the stretched segment begins.
            upper_bound (float): The upper bound of the grid.
            size1 (int): The number of elements in the linear segment.
            size2 (int): The number of elements in the stretched segment.
        """
        self.name = name
        self.type = CoordType.LINEAR_THEN_STRETCHED
        self.middle = middle
        self.size1 = size1
        self.size2 = size2

        # Create linear part
        linear_segment = np.linspace(lower_bound, middle, size1)
        dz = linear_segment[1] - linear_segment[0]

        # Create stretched part
        X = np.arange(size2 + 1)
        # Solve quadratic equation to match dz at transition and reach upper_bound
        alpha = 2.0 * ((upper_bound - middle) - size2 * dz) / (size2 * (size2 - 1.0))
        stretched_segment = middle + X * dz + 0.5 * X * (X - 1.0) * alpha

        self.grid = np.concatenate([linear_segment, stretched_segment[1:]])

    def write_hdf5(self, group: h5py.Group, dataset_name: str):
        """Write the coordinate data to an HDF5 group, including the middle value and segment sizes.

        Args:
            group (h5py.Group): The HDF5 group to write to.
            dataset_name (str): The name of the dataset.
        """
        super().write_hdf5(group, dataset_name)
        group.attrs["i_cut"] = self.size1
        group.attrs["z_cut"] = self.middle
