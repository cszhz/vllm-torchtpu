# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exception and result types of the megakernel library (framework-free)."""

from __future__ import annotations

from dataclasses import dataclass


class MegakernelConfigError(RuntimeError):
    """A requested configuration cannot be satisfied (e.g. a scratch budget overflow);
    never silently ignored.
    """


class MegakernelLayoutError(ValueError):
    """An externally provided cache layout is incompatible; callers may fall back to
    a non-megakernel path.
    """


@dataclass(frozen=True)
class EligibilityResult:
    """Structured eligibility result returned by startup and config gates."""

    eligible: bool
    reason: str = "ok"
    message: str = ""
    level: str = "warning"

    @property
    def code(self) -> str:
        return self.reason


__all__ = ["EligibilityResult", "MegakernelConfigError", "MegakernelLayoutError"]
