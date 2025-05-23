from typing import Any


class Record:
    """
    Record class to represent a record in the database.
    """

    api = None
    desc = None

    def __init__(self, api, desc: dict):
        self.api = api
        self.desc = desc

    def __str__(self) -> str:
        """
        Return a string description of the record.
        """
        user_names = []
        for user in self["users"]:
            user_names.append(user["firstname"] + " " + user["lastname"])
        return f"""Record(
    id={self['id']},
    date={self['date']},
    record_type={self['record_type']},
    decode_state={self['decode_state']},
    client_id={self['client_id']},
    group_id={self['group_id']},  # {self['group_name']}
    user_ids={self['user_ids']},  # {", ".join(user_names)}
    sport_id={self['sport_id']},  # {self['sport_disp_name']}
    device_id={self['device_id']},  # {self['device_name']}
    exercise_name={self['exercise_name']},
    size={self['size']},
)"""

    def __repr__(self) -> str:
        """
        Return a string representation of the record.
        """
        return f"Record(id={self['id']}, size={self['size']})"

    def __getitem__(self, key) -> Any:
        """
        Get the value of a key in the record.
        """
        if key in self.desc:
            return self.desc[key]
        else:
            raise KeyError(f"{key} not found in record.")

    def to_dict(self) -> dict:
        """
        Convert the record to a dictionary.
        """
        return self.desc
