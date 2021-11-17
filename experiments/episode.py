class Episode:
    def __init__(self, num_captures, captures_timesteps):
        self.num_captures = num_captures
        self.captures_timesteps = captures_timesteps
        self.first_capture_timestep = (
            captures_timesteps[0] if len(captures_timesteps) > 0 else None
        )
