"""
Composite Transformation
============================================
Multiple transformations can be used by providing a list of ``Transformation`` to ``CompositeTransformation``

"""

from textattack.shared import utils
from textattack.transformations import Transformation


class CompositeTransformation(Transformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.

    Args:
        transformations: The list of ``Transformation`` to apply.
    """

    def __init__(self, transformations, max_trans_num=5):
        if not (
            isinstance(transformations, list) or isinstance(transformations, tuple)
        ):
            raise TypeError("transformations must be list or tuple")
        elif not len(transformations):
            raise ValueError("transformations cannot be empty")
        self.transformations = transformations
        self.max_trans_num=max_trans_num

    def _get_transformations(self, *_):
        """Placeholder method that would throw an error if a user tried to
        treat the CompositeTransformation as a 'normal' transformation."""
        raise RuntimeError(
            "CompositeTransformation does not support _get_transformations()."
        )

    def __call__(self, *args, **kwargs):
        new_attacked_texts = set()
        for transformation in self.transformations:
            new_attacked_texts.update(transformation(*args, **kwargs))
        if 'return_indices' in kwargs:
            new_attacked_texts=list(new_attacked_texts)
        else:
            new_attacked_texts=list(new_attacked_texts)
        return new_attacked_texts[:self.max_trans_num]

    def __repr__(self):
        main_str = "CompositeTransformation" + "("
        transformation_lines = []
        for i, transformation in enumerate(self.transformations):
            transformation_lines.append(utils.add_indent(f"({i}): {transformation}", 2))
        transformation_lines.append(")")
        main_str += utils.add_indent("\n" + "\n".join(transformation_lines), 2)
        return main_str

    __str__ = __repr__
