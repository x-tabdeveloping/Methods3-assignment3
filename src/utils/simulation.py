import numpy as np
import pandas as pd

TRUE_EFFECTS = {
    "pitch_mode": 0.25,
    "pitch_variability": -0.55,
    "speech_rate": -0.75,
    "proportion_of_spoken_time": -1.25,
    "pause_number": 0.05,
    "pause_length": 1.89,
}

NO_EFFECT = {
    "pitch_mode": 0.0,
    "pitch_variability": 0.0,
    "speech_rate": 0.0,
    "proportion_of_spoken_time": 0.0,
    "pause_number": 0.0,
    "pause_length": 0.0,
}


def simulate_sample(
    n_group_size: int = 100, n_trials: int = 10, informed: bool = True
) -> pd.DataFrame:
    """Simulates a sample of participants.

    Parameters
    ----------
    n_group_size: int, default 100
        Number of participants in each group.
    n_trials: int, default 10
        Number of trials for each participant.
    informed: bool, default True
        Indicates whether the sample should be informed by former research or
        should just be random noise.

    Returns
    -------
    DataFrame
        Table of the simulated data.
    """
    if informed:
        effects = TRUE_EFFECTS
    else:
        effects = NO_EFFECT
    # Establishing the control individual effects of the pair
    print(
        {
            f"{effect_name}_mean": (-(effect / 2), 1)
            for effect_name, effect in effects.items()
        }
    )
    control = pd.DataFrame(
        {
            "participant_id": np.arange(n_group_size),
            "diagnosis": 0,
            **{
                f"{effect_name}_mean": np.random.normal(-effect / 2, 1, size=n_group_size)
                for effect_name, effect in effects.items()
            },
        }
    )
    schizophrenic = pd.DataFrame(
        {
            "participant_id": np.arange(n_group_size) + n_group_size,
            "diagnosis": 1,
            **{
                f"{effect_name}_mean": np.random.normal(effect / 2, 1, size=n_group_size)
                for effect_name, effect in effects.items()
            },
        }
    )
    # Concatenating the two groups
    participants = pd.concat((control, schizophrenic), ignore_index=True)
    # Repeating for each trial, and adding trial ids
    data = participants.loc[np.repeat(participants.index, n_trials)]
    data = data.assign(trial=np.tile(np.arange(n_trials), n_group_size * 2))
    trial_error = 0.5
    measurement_error = 0.2

    def measure(a):
        """
        Utility function for simulating measurement error and trial error.
        """
        return np.random.normal(
            np.random.normal(a, trial_error), measurement_error
        )

    # Collecting measurements
    data = data.assign(
        **{
            effect_name: measure(data[f"{effect_name}_mean"])
            for effect_name in effects
        }
    )
    n_observations = n_group_size * 2 * n_trials
    noise_variables = {
        f"noise{i_noise}": np.random.normal(0, 2, size=n_observations)
        for i_noise in range(4)
    }
    data = data.assign(**noise_variables)
    return data
