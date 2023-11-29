import numpy as np


def get_rasterized_maps(real_time_maps: list[np.ndarray], args: dict[str, any]) -> list[np.ndarray]:
    """
    Given real time Maps, calculates height Maps from the AFM tip 'point of view', i.e. according to its speed.
    The real time map resolution affects this, since for each pixel, the time floors to the most recent image.
    :param real_time_maps:
    :param args:
    :return:
    """
    if len(real_time_maps) <= 1:
        return []
    size_x = real_time_maps[0].shape[0]
    size_y = real_time_maps[0].shape[1]
    time_per_line, time_per_pixel = get_times(args, size_x)
    total_time = float(args["interval_ns"])
    if int(total_time / args["interval_ns"]) >= len(real_time_maps):
        return []
    if args["vertical_scanning"]:
        return vertical_scanning(args, real_time_maps, size_x, size_y, time_per_line, time_per_pixel, total_time)
    return horizontal_scanning(args, real_time_maps, size_x, size_y, time_per_line, time_per_pixel, total_time)


def horizontal_scanning(args, real_time_maps, size_x, size_y, time_per_line, time_per_pixel, total_time):
    rasterized_maps = []
    cur_rasterized_map_index = 0
    while total_time < args["simulation_end_time_ns"]:
        rasterized_maps.append(np.zeros(shape=real_time_maps[0].shape))
        for y in range(size_y):
            for x in range(size_x):
                rasterized_maps[cur_rasterized_map_index][x, y] = \
                    real_time_maps[int(total_time / args["interval_ns"])][
                        x, y]
                # Advance the time variable by the time it takes to capture a pixel.
                total_time += time_per_pixel
                if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                    break
            # Advance the time variable by the time it takes to move across a line.
            total_time += time_per_line
            if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                break
        # Advance the time variable by the time it takes to move back to the start.
        total_time += args["time_between_scans_ns"]
        if int(total_time / args["interval_ns"]) >= len(real_time_maps):
            break
        cur_rasterized_map_index += 1
    return rasterized_maps[:cur_rasterized_map_index]


def vertical_scanning(args, real_time_maps, size_x, size_y, time_per_line, time_per_pixel, total_time):
    rasterized_maps = []
    cur_rasterized_map_index = 0
    while total_time < args["simulation_end_time_ns"]:
        rasterized_maps.append(np.zeros(shape=real_time_maps[0].shape))
        for x in range(size_x):
            for y in range(size_y):
                rasterized_maps[cur_rasterized_map_index][x, y] = \
                    real_time_maps[int(total_time / args["interval_ns"])][
                        x, y]
                # Advance the time variable by the time it takes to capture a pixel.
                total_time += time_per_pixel
                if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                    break
            # Advance the time variable by the time it takes to move across a line.
            total_time += time_per_line
            if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                break
        # Advance the time variable by the time it takes to move back to the start.
        total_time += args["time_between_scans_ns"]
        if int(total_time / args["interval_ns"]) >= len(real_time_maps):
            break
        cur_rasterized_map_index += 1
    return rasterized_maps[:cur_rasterized_map_index]


def get_times(args: dict[str, any], size_x: int) -> tuple[float, float]:
    """
    time_per_pixel_ns, and time_per_line_ns are mutually exclusive arguments - here we calculate one from the
    other.
    :param args: User arguments.
    :param size_x: Size of the AFM image on the X axis.
    :return: tuple[time it takes the AFM to move across a line, time it takes the AFM to move across a pixel]
    """
    if args["time_per_line_ns"] is not None:
        time_per_line = args["time_per_line_ns"]
        time_per_pixel = args["time_per_line_ns"] / size_x
    else:  # args["time_per_pixel_ns"] is not None
        time_per_line = args["time_per_pixel_ns"] * size_x
        time_per_pixel = args["time_per_pixel_ns"]
    return time_per_line, time_per_pixel


def get_times_array(size_x, size_y, time_per_line, time_per_pixel, reset_time, raster_i):
    times = np.zeros(shape=(size_x, size_y))
    total_time = time_per_pixel + (time_per_line * size_y * 2 + reset_time) * raster_i
    for y in range(size_y):
        for x in range(size_x):
            times[x, y] = total_time
            # Advance the time variable by the time it takes to capture a pixel.
            total_time += time_per_pixel
        # Advance the time variable by the time it takes to move across a line.
        total_time += time_per_line
    return times


def modulo_raster(real_time_maps, args, max_images):
    modulo_maps = []
    for _ in range(max_images):
        modulo_maps.append(get_rasterized_maps(real_time_maps, args)[0])
        first = real_time_maps[0]
        real_time_maps = real_time_maps[1:]
        real_time_maps.append(first)
    return modulo_maps
