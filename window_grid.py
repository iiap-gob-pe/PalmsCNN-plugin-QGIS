# -*- coding: utf-8 -*-
"""Sliding-window geometry.

The arithmetic in :func:`window_centers` mirrors what
``palmeras_algo/apply_model.py`` does when it builds ``collist`` and ``rowlist``.
If one changes, the other stops describing what the model will actually do.

Nothing here reads pixel data: it only reports the grid a given pair of radii
produces, so the algorithm can log it before spending minutes on the models.
"""


def resolve_internal_radius(window_radius, internal_window_radius=None, factor=0.75):
    """Return the effective internal_window_radius, clamped to [1, window_radius].

    A null or zero value means "auto": the same formula
    ``palmeras_deteccion.apply_palmeras`` uses, ``int(round(window_radius * 0.75))``.
    """
    window_radius = int(window_radius)
    if not internal_window_radius:
        internal_window_radius = int(round(window_radius * factor))
    internal_window_radius = int(internal_window_radius)
    return max(1, min(window_radius, internal_window_radius))


def window_centers(size, window_radius, internal_window_radius):
    """Window centres along an axis of ``size`` pixels.

    Exact replica of apply_model.py::

        collist = list(range(wr, width - wr + 1, iwr * 2))
        if collist and collist[-1] < width - wr:
            collist.append(width - wr)
    """
    centers = list(range(window_radius, size - window_radius + 1,
                         internal_window_radius * 2))
    if centers and centers[-1] < size - window_radius:
        centers.append(size - window_radius)
    return centers


def _union_length(centers, radius, limit):
    """Length covered by the intervals [c-radius, c+radius], clipped to [0, limit]."""
    if not centers:
        return 0
    spans = sorted((max(0, c - radius), min(limit, c + radius)) for c in centers)
    total = 0
    start, end = spans[0]
    for lo, hi in spans[1:]:
        if lo <= end:
            end = max(end, hi)
        else:
            total += end - start
            start, end = lo, hi
    return total + (end - start)


def grid_geometry(width, height, window_radius, internal_window_radius, batch_size=0):
    """Describe the grid the sliding window will produce over a given raster."""
    wr = int(window_radius)
    iwr = resolve_internal_radius(wr, internal_window_radius)
    collist = window_centers(width, wr, iwr)
    rowlist = window_centers(height, wr, iwr)

    covered = (_union_length(collist, iwr, width) *
               _union_length(rowlist, iwr, height))
    total = float(width) * float(height)
    # Each batch is N windows of 2wr x 2wr x 3 bands in float32, the type the ONNX
    # model declares. With batch_size = 0 the batch is the whole column.
    per_batch = min(batch_size, len(rowlist)) if batch_size else len(rowlist)
    batch_mb = per_batch * (2 * wr) ** 2 * 3 * 4 / (1024.0 * 1024.0)

    return {
        'window_radius': wr,
        'internal_window_radius': iwr,
        'margin': wr - iwr,
        'stride': iwr * 2,
        'overlap': max(0, 2 * wr - 2 * iwr),
        'collist': collist,
        'rowlist': rowlist,
        'n_cols': len(collist),
        'n_rows': len(rowlist),
        'n_windows': len(collist) * len(rowlist),
        'coverage': (covered / total * 100.0) if total else 0.0,
        'batch_size': per_batch,
        'batch_mb': batch_mb,
        'raster_mb': total * 3 * 4 / (1024.0 * 1024.0),
    }


def describe_geometry(geo, width, height):
    """Readable summary of the grid, for the Processing log."""
    wr, iwr = geo['window_radius'], geo['internal_window_radius']
    lines = [
        'Raster: {} x {} px'.format(width, height),
        'window_radius = {} px  ->  window of {} x {} px',
        'internal_window_radius = {} px  ->  core of {} x {} px',
        'discarded margin = {} px per side',
        'stride = {} px, overlap between windows = {} px',
        'grid: {} rows x {} columns = {} windows',
        'inferences (one per column): {}',
        'batch per inference: {} windows, {:.1f} MB as float32',
        'normalised raster held in memory: {:.0f} MB (float32, loaded whole)',
        'core coverage: {:.1f} % of the raster',
    ]
    lines[1] = lines[1].format(wr, wr * 2, wr * 2)
    lines[2] = lines[2].format(iwr, iwr * 2, iwr * 2)
    lines[3] = lines[3].format(geo['margin'])
    lines[4] = lines[4].format(geo['stride'], geo['overlap'])
    lines[5] = lines[5].format(geo['n_rows'], geo['n_cols'], geo['n_windows'])
    lines[6] = lines[6].format(geo['n_cols'])
    lines[7] = lines[7].format(geo['batch_size'], geo['batch_mb'])
    lines[8] = lines[8].format(geo['raster_mb'])
    lines[9] = lines[9].format(geo['coverage'])

    if not geo['collist'] or not geo['rowlist']:
        lines.append('WARNING: 2 x window_radius is larger than the raster; '
                     'no window would be produced.')
    return lines
