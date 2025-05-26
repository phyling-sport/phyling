from typing import Any


class User:
    """
    User class to represent a user in the database.
    """

    api = None
    desc = {}

    def __init__(self, api, desc: dict):
        self.api = api
        self.desc = desc

    def __str__(self) -> str:
        """
        Return a string description of the user.
        """
        return f"""User(
    id={self['id']},
    mail={self['mail']},
    firstname={self['firstname']},
    lastname={self['lastname']},
)"""

    def __repr__(self) -> str:
        """
        Return a string representation of the user.
        """
        return f"User(id={self['id']}, mail={self['mail']})"

    def __getitem__(self, key) -> Any:
        """
        Get the value of a key in the user.
        """
        if key in self.desc:
            return self.desc[key]
        else:
            raise KeyError(f"{key} not found in user.")

    def to_dict(self) -> dict:
        """
        Convert the user to a dictionary.
        """
        return self.desc
