from collections.abc import Iterable
from dataclasses import dataclass, field
from types import EllipsisType as ellipsis
from typing import (
    Annotated,
    Any,
    Callable,
    Collection,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray

OtherSliceDef = tuple[(int | Callable[[int], int]), ...]
"""For a particular index of one gridsearch axis, which indexes of other axes
should be included."""
SkinnySpecs = tuple[tuple[str, ...], tuple[OtherSliceDef, ...]]


KeepAxisSpec = Union[
    tuple[ellipsis, ellipsis],  # all axes, all indices
    tuple[ellipsis, tuple[int, ...]],  # all axes, specific indices
    tuple[tuple[str, ...], ellipsis],  # specific axes, all indices
    Collection[tuple[str, tuple[int, ...]]],  # specific axes, specific indices
]


@dataclass(frozen=True)
class GridLocator:
    """A specification of which points in a gridsearch to match.

    Rather than specifying the exact point in the mega-grid of every
    varied axis, specify by result, e.g "all of the points from the
    Kalman series that had the best mean squared error as noise was
    varied.

    Logical AND is applied across the metric, keep_axis, AND param_match
    specifications.

    Args:
        metric: The metric in which to find results.  An ellipsis means "any
            metrics"
        keep_axis: The grid-varied parameter in which to find results and which
            index of values for that parameter.  To search a particular value of
            that parameter, use the param_match kwarg.It can be specified in
            several ways:
            (a) a tuple of two ellipses, representing all axes, all indices
            (b) a tuple of an ellipsis, representing all axes, and a tuple of
                ints for specific indices
            (c) a tuple of a tuple of strings for specific axes, and an ellipsis
                for all indices
            (d) a collection of tuples of a string (specific axis) and tuple of
                ints (specific indices)
        param_match: A collection of dictionaries to match parameter values represented
            by points in the gridsearch.  Dictionary equality is checked for every
            non-callable value; for callable values, it is applied to the grid
            parameters and must return a boolean.  For values whose equality is object
            equality (often, mutable objects), the repr is used.  Logical OR is applied
            across the collection.
    """

    metrics: Collection[str] | ellipsis = field(default=...)
    keep_axes: KeepAxisSpec = field(default=(..., ...))
    params_or: Collection[dict[str, Any]] = field(default=())


T = TypeVar("T", bound=np.generic)
GridsearchResult = Annotated[NDArray[T], "(n_metrics, n_plot_axis)"]
SeriesData = Annotated[
    list[
        tuple[
            Annotated[GridsearchResult, "metrics"],
            Annotated[GridsearchResult[np.void], "arg_opts"],
        ]
    ],
    "len=n_plot_axes",
]

ExpResult = dict[str, Any]


class SavedGridPoint(TypedDict):
    """The results at a point in the gridsearch.

    Args:
        params: the full list of parameters identifying this variant
        pind: the full index in the series' grid
        data: the results of the experiment
    """

    params: dict
    pind: tuple[int, ...]
    data: ExpResult


class GridsearchResultDetails(TypedDict):
    system: str
    plot_data: list[SavedGridPoint]
    series_data: dict[str, SeriesData]
    metrics: tuple[str, ...]
    grid_params: list[str]
    grid_vals: list[Sequence[Any]]
    scan_grid: dict[str, Sequence[Any]]
    plot_grid: dict[str, Sequence[Any]]
    main: float


@dataclass
class SeriesDef:
    """The details of constructing the ragged axes of a grid search.

    The concept of a SeriesDef refers to a slice along a single axis of
    a grid search in conjunction with another axis (or axes)
    whose size or meaning differs along different slices.

    Attributes:
        name: The name of the slice, as a label for printing
        static_param: the constant parameter to this slice. Then key is
            the name of the parameter, as understood by the experiment
            Conceptually, the key serves as an index of this slice in
            the gridsearch.
        grid_params: the keys of the parameters in the experiment that
            vary along jagged axis for this slice
        grid_vals: the values of the parameters in the experiment that
            vary along jagged axis for this slice

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )

    """

    name: str
    static_param: dict
    grid_params: list[str]
    grid_vals: list[Iterable]


@dataclass
class SeriesList:
    """Specify the ragged slices of a grid search.

    As an example, consider a grid search of miles per gallon for
    different vehicles, in different routes, with different tires.
    Since different tires fit on different vehicles, the tire axis would
    be ragged, varying along the vehicle axis.

        Truck = SeriesDef("trucks")

    Attributes:
        param_name: the key of the parameter in the experiment that
            varies along the series axis.
        print_name: the print name of the parameter in the experiment
            that varies along the series axis.
        series_list: Each element of the series axis

    Example:

        truck_wheels = SeriesDef(
            "Truck",
            {"vehicle": "flatbed_truck"},
            ["vehicle.n_wheels"],
            [[10, 18]]
        )
        bike_tires = SeriesDef(
            "Bike",
            {"vehicle": "bicycle"},
            ["vehicle.tires"],
            [["gravel_tires", "road_tires"]]
        )
        VehicleOptions = SeriesList(
            "vehicle",
            "Vehicle Types",
            [truck_wheels, bike_tires]
        )

    """

    param_name: Optional[str]
    print_name: Optional[str]
    series_list: list[SeriesDef]
