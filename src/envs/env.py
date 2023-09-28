class Env:
    def step(self, action) -> tuple[dict, float | int]:
        raise NotImplementedError

    def reset(self, seed: int) -> dict:
        raise NotImplementedError

    def get_state(self) -> dict:
        raise NotImplementedError
