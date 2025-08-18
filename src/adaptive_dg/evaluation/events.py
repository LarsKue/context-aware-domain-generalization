from pathlib import Path

import numpy as np
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_events(path, version):
    path = Path(path)
    path = path / f"version_{version}"
    events = list(path.glob("events.out.tfevents.*.1"))[0]
    return events


def events_to_dataframe(events):
    accumulator = EventAccumulator(str(events))
    accumulator.Reload()

    max_step = accumulator.Scalars("epoch")[-1].step
    index = pd.Index(range(max_step + 1), name="step")
    columns = accumulator.Tags()["scalars"]

    data = {}

    for key in columns:
        steps, indices = np.unique([event.step for event in accumulator.Scalars(key)], return_index=True)

        values = [event.value for event in accumulator.Scalars(key)]
        values = np.array(values)[indices]

        series = pd.Series(data=values, index=steps, name=key)
        series = series.reindex(index)

        data[key] = series

    return pd.DataFrame(data, index=index, columns=columns)
