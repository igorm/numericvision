"""Classes and functions for analyzing image contours and detecting sequences of seven-segment digit boxes."""
import math
import numpy as np
import cv2
import numericvision as nv


class Bag:
    """Encapsulates logic which analyzes input image contours and identifies sequences of seven-segment digit boxes."""

    def __init__(self, image, contours, hierarchy, roi_contour=None):
        self.image = image
        self.contours = contours
        self.hierarchy = hierarchy
        self.roi_contour = roi_contour

        self.boxes = []
        self.sequences = []

        self._set_boxes()
        self._merge_shards()
        self._remove_duplicates()
        self._set_sequences()
        self._remove_subsequences()

    def _set_boxes(self):
        """Instantiates a Box for each contour, filters out Boxes which don't meet seven-segment digit criteria."""
        keys_to_boxes = {}
        for points, node in zip(self.contours, self.hierarchy):
            area = cv2.contourArea(points)
            perimeter = cv2.arcLength(points, True)
            if area < 1 or perimeter < 1:
                continue

            box = Box(points)
            image_area = float(self.image.shape[1] * self.image.shape[0])
            is_box_in_roi = (
                self.roi_contour.contains_point(box.contour.c_point) if self.roi_contour else True
            )
            if (
                box.key not in keys_to_boxes
                and box.contour.area / image_area * 100 > nv.BOX_MIN_IMAGE_AREA_PCT
                and box.contour.aspect_ratio > nv.BOX_MIN_ASPECT_RATIO
                and box.contour.aspect_ratio < nv.BOX_MAX_ASPECT_RATIO
                and is_box_in_roi
            ):
                keys_to_boxes[box.key] = box

        if keys_to_boxes:
            keys = np.array(list(keys_to_boxes.keys()), dtype=[("x", int), ("y", int)])
            indices = np.lexsort((keys["x"], keys["y"]))  # tb, lr
            box_keys = list(map(tuple, keys[indices]))
            self.boxes = [keys_to_boxes[k] for k in box_keys]

    def _merge_shards(self):
        """Identifies Boxes representing parts of the same seven-segment digit and merges them together."""
        shard_box_keys = []
        for box in self.boxes:
            if box.key in shard_box_keys:
                continue

            shard_box = next(
                (b for b in self.boxes if (b.key != box.key and b.is_shard_of(box))), None,
            )
            if shard_box:
                box.merge(shard_box)
                shard_box_keys.append(shard_box.key)

        self.boxes = [b for b in self.boxes if b.key not in shard_box_keys]

    def _remove_duplicates(self):
        """Removes duplicate Boxes."""
        duplicate_box_keys = []
        for box in self.boxes:
            if box.key in duplicate_box_keys:
                continue

            for duplicate_box in (
                b for b in self.boxes if (b.key != box.key and b.is_duplicate_of(box))
            ):
                duplicate_box_keys.append(duplicate_box.key)

        self.boxes = [b for b in self.boxes if b.key not in duplicate_box_keys]

    def _set_sequences(self):
        """Identifies Sequences of Boxes."""
        for box in self.boxes:
            next_box = next(
                (b for b in self.boxes if (b.key != box.key and b.is_next_after(box))), None,
            )
            if next_box:
                box.next = next_box

        for box in (b for b in self.boxes if b.next):
            box.set_sequence_boxes()

        for box in sorted(self.boxes, key=lambda b: b.get_sequence_box_count(), reverse=True):
            if (
                box.get_sequence_box_count() < nv.SEQ_MIN_BOX_COUNT
                # Any of the boxes is already claimed by another sequence
                or next((b for b in box.sequence_boxes if b.sequence), None)
            ):
                continue

            sequence = Sequence(box.sequence_boxes)
            self.sequences.append(sequence)

        # b, x, b, b
        for box in (b for b in self.boxes if not b.sequence):
            for sequence in (s for s in self.sequences if s.get_box_count() == 2):
                if sequence.boxes[0].is_next_after(box, True):
                    sequence.patch_prepend_box(box)

        # b, b, x, b
        for sequence in (s for s in self.sequences if s.get_box_count() == 2):
            for box in (b for b in self.boxes if not b.sequence):
                if box.is_next_after(sequence.boxes[-1], True):
                    sequence.patch_append_box(box)

        self.sequences = [
            s
            for s in self.sequences
            if (
                s.get_box_count() > nv.SEQ_MIN_BOX_COUNT
                and s.get_avg_distance_to_center_line() < nv.SEQ_MAX_AVG_DISTANCE_TO_CENTER_LINE
                and s.get_x_d_min_max_d_pct() < nv.SEQ_MAX_X_D_MIN_MAX_D_PCT
                and s.get_h_min_max_d_pct() < nv.SEQ_MAX_H_MIN_MAX_D_PCT
            )
        ]

    def _remove_subsequences(self):
        """Removes overlapping Sequences."""
        subsequence_keys = []
        for sequence in self.sequences:
            if sequence.key in subsequence_keys:
                continue

            for subsequence in (
                s
                for s in self.sequences
                if (s.key != sequence.key and s.is_subsequence_of(sequence))
            ):
                subsequence_keys.append(subsequence.key)

        self.sequences = [b for b in self.sequences if b.key not in subsequence_keys]

    def get_box_count(self):
        """The Box count."""
        return len(self.boxes)

    def get_sequence_count(self):
        """The Sequence count."""
        return len(self.sequences)


class Sequence:
    """Represents a sequence of seven-segment digit boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        for box in boxes:
            box.sequence = self

        self.key = self.get_contour().c_point
        self.patched_box_count = 0

    def get_top_line(self):
        """A line segment drawn through extreme top points of the Sequence's first and last Boxes."""
        line = (self.boxes[0].source_contour.et_point, self.boxes[-1].source_contour.et_point)
        return extend_line(line)

    def get_bottom_line(self):
        """A line segment drawn through extreme bottom points of the Sequence's first and last Boxes."""
        line = (self.boxes[0].source_contour.eb_point, self.boxes[-1].source_contour.eb_point)
        return extend_line(line)

    def get_left_line(self):
        """The left vertical line of the Sequence's first Box."""
        return self.boxes[0].get_left_vertical_line()

    def get_right_line(self):
        """The right vertical line of the Sequence's last Box."""
        return self.boxes[-1].get_right_vertical_line()

    def get_contour(self):
        """A Sequence's contour is a tetragon drawn around the Sequence's Boxes."""
        top_line = self.get_top_line()
        right_line = self.get_right_line()
        bottom_line = self.get_bottom_line()
        left_line = self.get_left_line()

        return Tetragon(
            get_intersection_point(top_line, left_line),
            get_intersection_point(top_line, right_line),
            get_intersection_point(bottom_line, right_line),
            get_intersection_point(bottom_line, left_line),
        )

    def get_padded_contour(self, padding):
        """Pads the Sequence's contour."""
        contour = self.get_contour()

        return Tetragon(
            (contour.tl_point[0] - padding, contour.tl_point[1] - padding),
            (contour.tr_point[0] + padding, contour.tr_point[1] - padding),
            (contour.br_point[0] + padding, contour.br_point[1] + padding),
            (contour.bl_point[0] - padding, contour.bl_point[1] + padding),
        )

    def get_box_count(self):
        """The Box count."""
        return len(self.boxes) + self.patched_box_count

    def patch_prepend_box(self, box, patched_box_count=1):
        """Prepends a Box when patching a Sequence with a missing Box: box missing box box."""
        box.sequence = self
        self.boxes.insert(0, box)
        self.patched_box_count += patched_box_count

    def patch_append_box(self, box, patched_box_count=1):
        """Appends a Box when patching a Sequence with a missing Box: box box missing box."""
        box.sequence = self
        self.boxes.append(box)
        self.patched_box_count += patched_box_count

    def get_x_ds(self):
        """A list of Boxes' center point x deltas."""
        x_ds = []
        for previous_box, box in zip(self.boxes, self.boxes[1:]):
            a_contour = previous_box.backbone_contour
            b_contour = box.backbone_contour
            x_ds.append(b_contour.c_point[0] - a_contour.c_point[0])

        return x_ds

    def get_avg_x_d(self):
        """The average Boxes' center point x delta."""
        x_ds = self.get_x_ds()

        return sum(x_ds) / float(len(x_ds))

    def get_x_d_min_max_d_pct(self):
        """Percentage difference between max and min Boxes' center point x deltas."""
        x_ds = self.get_x_ds()

        if self.patched_box_count > 0:
            x_ds.sort()
            max_x_d = x_ds.pop(-1)
            x_ds.append(max_x_d / 2)
            x_ds.append(max_x_d / 2)

        return get_d_pct(max(x_ds), min(x_ds))

    def get_hs(self):
        """Heights of the Sequence's Boxes."""
        return [b.backbone_contour.h for b in self.boxes]

    def get_h_min_max_d_pct(self):
        """Percentage difference between max and min Boxes' heights."""
        hs = self.get_hs()

        return get_d_pct(max(hs), min(hs))

    def get_avg_distance_to_center_line(self):
        """Average distance from Boxes' center points to the Sequence's center line."""
        distances = []
        for box in self.boxes[1:-1]:
            distance = get_point_to_line_distance(
                box.contour.c_point,
                (self.boxes[0].contour.c_point, self.boxes[-1].contour.c_point),
            )
            distances.append(distance)

        return sum(distances) / float(len(distances))

    def is_subsequence_of(self, sequence):
        """Is overlapping with the provided Sequence?"""
        return sequence.get_contour().contains_point(self.get_contour().c_point)


class Box:
    """Represents a seven-segment digit box."""

    def __init__(self, points):
        self.key = None
        self.is_tall_narrow = False

        self.source_contour = None
        self.contour = None
        self.backbone_contour = None
        self.tl_contour = None
        self.t_contour = None
        self.tr_contour = None
        self.c_contour = None
        self.br_contour = None
        self.b_contour = None
        self.bl_contour = None

        self.next = None
        self.sequence_boxes = []
        self.sequence = None

        self._set_contours(points)

    def _set_contours(self, points, tl_point=None, w=0, h=0):
        """Sets the Box's contours based on provided points."""
        if tl_point is None:
            x, y, w, h = cv2.boundingRect(points)
            tl_point = x, y

        if self.source_contour is not None:
            points = np.concatenate((self.source_contour.points, points))

        self.source_contour = Polygon(points)
        self.contour = Rectangle(tl_point, w, h)
        self.key = self.contour.c_point

        if self.contour.aspect_ratio > nv.BOX_MAX_TALL_NARROW_ASPECT_RATIO:
            x, y = tl_point

            self.backbone_contour = Rectangle((x + 2 / 3 * w, y), 1 / 3 * w, h)
            self.tl_contour = Rectangle((x, y + 1 / 7 * h), 1 / 3 * w, 2 / 7 * h)
            self.t_contour = Rectangle((x + 1 / 3 * w, y), 1 / 3 * w, 1 / 7 * h)
            self.tr_contour = Rectangle((x + 2 / 3 * w, y + 1 / 7 * h), 1 / 3 * w, 2 / 7 * h)
            self.c_contour = Rectangle((x + 1 / 3 * w, y + 3 / 7 * h), 1 / 3 * w, 1 / 7 * h)
            self.br_contour = Rectangle((x + 2 / 3 * w, y + 4 / 7 * h), 1 / 3 * w, 2 / 7 * h)
            self.b_contour = Rectangle((x + 1 / 3 * w, y + 6 / 7 * h), 1 / 3 * w, 1 / 7 * h)
            self.bl_contour = Rectangle((x, y + 4 / 7 * h), 1 / 3 * w, 2 / 7 * h)
        else:
            self.backbone_contour = self.contour
            self.is_tall_narrow = True

    def _process_sequence_boxes(self, box):
        """Looks ahead left to right at Boxes following the current Box. If following Boxes meet "next criteria", adds
        them to sequence_boxes.
        """
        if not box:
            return

        self.sequence_boxes.append(box)
        self._process_sequence_boxes(box.next)

    def is_next_after(self, box, is_patching=False):
        """Does the current Box meet criteria to be considered next in a sequence after the provided Box?"""
        a_contour = box.backbone_contour
        b_contour = self.backbone_contour

        x_d = b_contour.c_point[0] - a_contour.c_point[0]
        y_d = abs(b_contour.c_point[1] - a_contour.c_point[1])
        w_d = abs(b_contour.w - a_contour.w)
        h_d = abs(b_contour.h - a_contour.h)

        min_x_d = 0
        max_x_d = 0.95 * a_contour.h

        if is_patching:
            # Should be twice the x_d of the sequence
            sequence = self.sequence if self.sequence else box.sequence
            expected_x_d = 2 * sequence.get_avg_x_d()
            min_x_d = expected_x_d - expected_x_d * 0.1
            max_x_d = expected_x_d + expected_x_d * 0.03

        max_y_d = 0.25 * a_contour.h
        max_w_d = 0.50 * a_contour.w
        max_h_d = 0.25 * a_contour.h

        is_next = (
            # Main
            # cv2.pointPolygonTest(box.contour.points, self.contour.c_point, False) < 0
            # and cv2.pointPolygonTest(self.contour.points, box.contour.c_point, False) < 0
            self.contour.tl_point[0] - box.contour.tr_point[0] >= 0  # Right margin
            # Backbone
            and (x_d > min_x_d and x_d < max_x_d)
            and y_d < max_y_d
            and w_d < max_w_d
            and h_d < max_h_d
        )

        # if self.key[0] == 262:
        #     print "(%i, %i) > (%i, %i) = %r" % (box.key[0], box.key[1], self.key[0], self.key[1], is_next)
        #     print "    x_d=%.2f [%.2f, %.2f]" % (x_d, min_x_d, max_x_d)
        #     print "    y_d=%.2f [%.2f]" % (y_d, max_y_d)
        #     print "    w_d=%.2f [%.2f]" % (w_d, max_w_d)

        return is_next

    def is_duplicate_of(self, box):
        """Does the current Box meet criteria to be considered a duplicate of the provided Box?"""
        a_contour = box.contour
        b_contour = self.contour

        x_d = abs(b_contour.c_point[0] - a_contour.c_point[0])
        y_d = abs(b_contour.c_point[1] - a_contour.c_point[1])
        w_d = abs(a_contour.w - b_contour.w)
        h_d = abs(a_contour.h - b_contour.h)

        return (
            x_d < nv.BOX_DUPLICATE_MAX_X_D
            and y_d < nv.BOX_DUPLICATE_MAX_Y_D
            and w_d < nv.BOX_DUPLICATE_MAX_W_D
            and h_d < nv.BOX_DUPLICATE_MAX_H_D
        )

    def is_shard_of(self, box):
        """Does the current Box meet criteria to be considered a shard of the provided Box?"""
        a_contour = box.contour
        b_contour = self.contour

        x_d = abs(b_contour.c_point[0] - a_contour.c_point[0])
        y_d = b_contour.c_point[1] - a_contour.c_point[1]
        w_d = abs(a_contour.w - b_contour.w)
        h_d = abs(a_contour.h - b_contour.h)

        return (
            x_d < nv.BOX_SHARD_MAX_X_D
            and y_d > nv.BOX_SHARD_MIN_Y_D
            and abs(a_contour.h - y_d) < nv.BOX_SHARD_MAX_H_Y_D
            and w_d < nv.BOX_SHARD_MAX_W_D
            and h_d < nv.BOX_SHARD_MAX_H_D
        )

    def merge(self, shard_box):
        """Merges contours of the current Box with contours of the shard."""
        x = min(self.contour.tl_point[0], shard_box.contour.bl_point[0])
        y = self.contour.tl_point[1]
        w = max(self.contour.tr_point[0], shard_box.contour.br_point[0]) - x
        h = shard_box.contour.bl_point[1] - self.contour.tl_point[1]

        self._set_contours(shard_box.source_contour.points, (x, y), w, h)

    def set_sequence_boxes(self):
        """Runs the look ahead process of identifying a sequence of "next Boxes"."""
        self._process_sequence_boxes(self)

    def get_sequence_box_count(self):
        """The count of "next Boxes" following the current Box."""
        return len(self.sequence_boxes)

    def get_left_vertical_line(self):
        """A line segment drawn through extreme left points of the Box's source contour."""
        line, i_point, e_point_to_line_distance = self.get_vertical_line_with_context()

        if i_point[0] < self.contour.c_point[0]:
            return line
        else:
            a_point, b_point = line

            return (
                (int(a_point[0] - e_point_to_line_distance), a_point[1]),
                (int(b_point[0] - e_point_to_line_distance), b_point[1]),
            )

    def get_right_vertical_line(self):
        """A line segment drawn through extreme right points of the Box's source contour."""
        line, i_point, e_point_to_line_distance = self.get_vertical_line_with_context()

        if i_point[0] > self.contour.c_point[0]:
            return line
        else:
            a_point, b_point = line

            return (
                (int(a_point[0] + e_point_to_line_distance), a_point[1]),
                (int(b_point[0] + e_point_to_line_distance), b_point[1]),
            )

    def get_vertical_line_with_context(self):
        """The extended vertical line segment with spacial context."""
        line = self.get_extended_vertical_line()
        e_point = sorted(
            self.source_contour.get_points(),
            key=lambda p: get_point_to_line_distance(p, line),
            reverse=True,
        )[0]

        return (
            line,
            get_intersection_point(line, self.contour.get_center_line()),
            get_point_to_line_distance(e_point, line),
        )

    def get_extended_vertical_line(self):
        """The longest vertical line segment extended to match the height of the Box's contour."""
        a_point, b_point = self.get_vertical_line()
        y_d = b_point[1] - a_point[1]
        extend_factor = (self.contour.h / y_d) * 2

        return extend_line((a_point, b_point), extend_factor)

    def get_vertical_line(self):
        """The longest vertical line segment drawn through the Box's source contour."""
        return sorted(
            self.source_contour.get_vertical_lines(),
            key=lambda l: abs(l[1][1] - l[0][1]),
            reverse=True,
        )[0]


class Polygon:
    """Represents a polygon contour."""

    def __init__(self, points):
        self.points = points

        self.el_point = tuple(points[points[:, :, 0].argmin()][0])
        self.et_point = tuple(points[points[:, :, 1].argmin()][0])
        self.er_point = tuple(points[points[:, :, 0].argmax()][0])
        self.eb_point = tuple(points[points[:, :, 1].argmax()][0])

        self.moments = cv2.moments(points)
        self.c_point = (
            (
                int(self.moments["m10"] / self.moments["m00"]),
                int(self.moments["m01"] / self.moments["m00"]),
            )
            if self.moments["m00"] != 0
            else None
        )

        self.w = self.er_point[0] - self.el_point[0]
        self.h = self.eb_point[1] - self.et_point[1]
        self.aspect_ratio = self.w / float(self.h) if self.h > 0 else 0
        self.area = cv2.contourArea(points)
        self.perimeter = cv2.arcLength(points, True)

    def contains_point(self, point):
        """Is the provided point inside of the Polygon?"""
        return cv2.pointPolygonTest(self.points, point, False) > 0

    def contains_contour(self, contour):
        """Is the provided contour inside of the Polygon?"""
        return (
            self.contains_point(contour.el_point)
            and self.contains_point(contour.et_point)
            and self.contains_point(contour.er_point)
            and self.contains_point(contour.eb_point)
        )

    def get_vertical_lines(self):
        """Vertical line segments drawn throug the Polygon's points."""
        points = self.get_points()
        vertical_segments = []

        for i, a_point in enumerate(points):
            if i + 2 > len(points):
                break

            if vertical_segments and a_point in vertical_segments[-1]:
                continue

            segment = [a_point]

            for b_point in points[i + 2 :]:
                y_d = abs(b_point[1] - a_point[1])
                if (vertical_segments and b_point in vertical_segments[-1]) or y_d < 5:
                    break

                j = points.index(b_point)

                is_segment_straight = True
                intermediate_points = [points[j - 1]] if i + 1 == j - 1 else points[i:j]

                for point in intermediate_points:
                    if get_point_to_line_distance(point, (a_point, b_point)) > 1:
                        is_segment_straight = False
                        break

                if is_segment_straight:
                    segment.append(b_point)
                else:
                    break

            if len(segment) > 1:
                vertical_segments.append((segment[0], segment[-1]))

        return vertical_segments

    def get_points(self):
        """The points of the Polygon's contour."""
        return [tuple(p[0]) for p in self.points]


class Tetragon(Polygon):
    """Represents a tetragon contour. Extends Polygon."""

    def __init__(self, tl_point, tr_point, br_point, bl_point):
        super(Tetragon, self).__init__(
            np.array([[tl_point], [tr_point], [br_point], [bl_point],], dtype=np.int32)
        )

        self.tl_point = tl_point
        self.tr_point = tr_point
        self.br_point = br_point
        self.bl_point = bl_point

    def get_center_line(self):
        """The horizontal center line segment."""
        return (
            (self.tl_point[0], self.c_point[1]),
            (self.tr_point[0], self.c_point[1]),
        )


class Rectangle(Tetragon):
    """Represents a rectangular contour. Extends Tetragon."""

    def __init__(self, tl_point, w, h):
        x, y = tl_point

        super(Rectangle, self).__init__((x, y), (x + w, y), (x + w, y + h), (x, y + h))

    @classmethod
    def from_tl_point_br_point(cls, tl_point, br_point):
        """A factory method for creating Rectangles based on TL and BR points."""
        return cls(tl_point, br_point[0] - tl_point[0], br_point[1] - tl_point[1])


class Line:
    """Represents a line segment."""

    def __init__(self, a_point, b_point):
        self.a_point = a_point
        self.b_point = b_point

    def __str__(self):
        return "Line(a={},b={})".format(self.a_point, self.b_point)

    def get_a_b_points(self):
        """The Line's A, B points."""
        return self.a_point, self.b_point

    def get_length(self):
        """The Line's length."""
        return (
            (self.a_point[0] - self.b_point[0]) ** 2 + (self.a_point[1] - self.b_point[1]) ** 2
        ) ** 0.5

    def extend(self, factor):
        """Extends the Line by the provided factor.
        http://stackoverflow.com/questions/28825461/how-to-extend-a-line-segment-in-both-directions
        """
        x_d = self.b_point[0] - self.a_point[0]
        a_point_x = self.a_point[0] - (x_d * (factor - 1) / 2)
        b_point_x = self.b_point[0] + (x_d * (factor - 1) / 2)

        y_d = self.b_point[1] - self.a_point[1]
        a_point_y = self.a_point[1] - (y_d * (factor - 1) / 2)
        b_point_y = self.b_point[1] + (y_d * (factor - 1) / 2)

        self.a_point = (a_point_x, a_point_y)
        self.b_point = (b_point_x, b_point_y)

        return self

    def get_intersection_point_with(self, line):
        """The intersection point of this Line and the provided Line.
        http://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
        """
        xdiff = (self.a_point[0] - self.b_point[0], line.a_point[0] - line.b_point[0])
        ydiff = (self.a_point[1] - self.b_point[1], line.a_point[1] - line.b_point[1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception("Lines do not intersect")

        d = (det(self.a_point, self.b_point), det(line.a_point, line.b_point))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        return x, y

    def get_distance_to_point(self, point):
        """Distance from the provided point to this Line.
        https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment
        """
        px, py = point
        x1, y1 = self.a_point
        x2, y2 = self.b_point

        line_mag = math.hypot(x2 - x1, y2 - y1)

        if line_mag < 0.00000001:
            return 9999

        u1 = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))
        u = u1 / (line_mag * line_mag)

        if (u < 0.00001) or (u > 1):
            ix = math.hypot(x1 - px, y1 - py)
            iy = math.hypot(x2 - px, y2 - py)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = math.hypot(ix - px, iy - py)

        return distance


def extend_line(line, factor=2):
    """Extends the line segment by the provided factor."""
    return Line(*line).extend(factor).get_a_b_points()


def get_intersection_point(a_line, b_line):
    """The intersection point of two line segments."""
    return Line(*a_line).get_intersection_point_with(Line(*b_line))


def get_point_to_line_distance(point, line):
    """Distance from the provided point to the line segment."""
    return Line(*line).get_distance_to_point(point)


def get_d_pct(a, b):
    """Percentage difference between two values."""
    d = float(abs(a - b))
    avg = float((a + b) / 2)

    return round(d / avg * 100)
