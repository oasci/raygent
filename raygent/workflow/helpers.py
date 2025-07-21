# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import TypeVar, override

from raygent import Task

T = TypeVar("T")


class IdentityTask(Task):
    """Return the data unchanged."""

    @override
    def do(self, payload: T) -> T:
        return payload
