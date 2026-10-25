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
from unittest.mock import MagicMock, patch

from vllm_torchtpu import utils


class _Libc:
    """Stand-in for ctypes.CDLL(None): only the symbols given exist."""

    def __init__(self, **symbols):
        self.__dict__.update(symbols)


def test_tcmalloc_page_heap_is_released():
    release, trim = MagicMock(), MagicMock()
    libc = _Libc(MallocExtension_ReleaseFreeMemory=release, malloc_trim=trim)
    with patch.object(utils.ctypes, "CDLL", return_value=libc):
        utils.release_memory_to_os()
    release.assert_called_once_with()
    trim.assert_not_called()


def test_glibc_arenas_are_trimmed_without_tcmalloc():
    trim = MagicMock()
    with patch.object(utils.ctypes, "CDLL", return_value=_Libc(malloc_trim=trim)):
        utils.release_memory_to_os()
    trim.assert_called_once_with(0)


def test_an_allocator_with_neither_entry_point_is_skipped_with_a_warning():
    with (
        patch.object(utils.ctypes, "CDLL", return_value=_Libc()),
        patch.object(utils.logger, "warning") as warning,
    ):
        utils.release_memory_to_os()
    warning.assert_called_once()


def test_release_logs_rss_before_and_after():
    trim = MagicMock()
    with (
        patch.object(utils.ctypes, "CDLL", return_value=_Libc(malloc_trim=trim)),
        patch.object(utils, "host_rss_gib", side_effect=[80.5, 2.8]),
        patch.object(utils.logger, "info") as info,
    ):
        utils.release_memory_to_os()
    info.assert_called_once()
    assert info.call_args.args[1:] == (80.5, 2.8)
