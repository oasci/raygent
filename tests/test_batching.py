# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scientific Computing Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from raygent.batch import batch_generator


def test_batch_generator():
    """
    Checks that the task generator works correctly based on batch size.
    """
    batch = list(range(10))
    batch_size = 3
    batches = list(batch_generator((batch,), batch_size=batch_size))

    assert len(batches) == 4
    assert batches[0][0] == 0
    assert batches[0][1] == ([0, 1, 2],)
    assert batches[1][0] == 1
    assert batches[1][1] == ([3, 4, 5],)
    assert batches[2][0] == 2
    assert batches[2][1] == ([6, 7, 8],)
    assert batches[3][0] == 3
    assert batches[3][1] == ([9],)


def test_batch_generator_large_batch():
    """
    Checks that only one batch is returned when batch > len(batch)
    """
    batch = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 100
    batches = list(batch_generator((batch,), batch_size=batch_size))
    assert len(batches) == 1
    assert batches[0][0] == 0
    assert batches[0][1] == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],)


def test_batch_generator_multiple_batches():
    batch1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch2 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    batch_size = 3
    batches = list(batch_generator((batch1, batch2), batch_size=batch_size))
    assert len(batches) == 4
    assert batches[0][0] == 0
    assert batches[0][1] == ([0, 1, 2], ["A", "B", "C"])
