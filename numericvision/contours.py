import numpy as np
import cv2
import math


class Bag(object):

    def __init__(self, image, contours, hierarchy, roi_contour=None):
        self.image = image
        self.contours = contours
        self.hierarchy = hierarchy
        self.roi_contour = roi_contour

        self.keys_to_boxes = {}
        self.box_keys = []
        self.boxes = []
        self.sequences = []

        self._set_boxes()
        self._merge_shards()
        self._remove_duplicates()
        self._set_sequences()
        self._remove_subsequences()

    def _get_image_area(self):
        return self.image.shape[1] * self.image.shape[0]

    def _set_boxes(self):
        for points, node in zip(self.contours, self.hierarchy):
            area = cv2.contourArea(points)
            perimeter = cv2.arcLength(points, True)
            if area < 1 or perimeter < 1:
                continue

            box = Box(points)
            is_box_in_roi = self.roi_contour.contains_point(box.contour.c_point) if self.roi_contour else True
            if (box.key not in self.keys_to_boxes
                and box.contour.area / float(self._get_image_area()) * 100 > 0.01
                and box.contour.aspect_ratio > 0.1
                and box.contour.aspect_ratio < 0.8
                and is_box_in_roi
            ):
                self.keys_to_boxes[box.key] = box

        if self.keys_to_boxes:
            keys = np.array(
                list(self.keys_to_boxes.keys()),
                dtype=[('x', int), ('y', int)]
            )
            indices = np.lexsort((keys['x'], keys['y'])) # tb, lr
            self.box_keys = list(map(tuple, keys[indices]))
            self.boxes = [self.keys_to_boxes[k] for k in self.box_keys]

    def _set_sequences(self):
        for box in self.boxes:
            next_box = next((b for b in self.boxes if (
                b.key != box.key
                and b.is_next_after(box)
            )), None)
            if next_box:
                box.next = next_box

        for box in (b for b in self.boxes if b.next):
            box.set_sequence_boxes()

        for box in sorted(self.boxes, key=lambda b: b.get_sequence_box_count(), reverse=True):
            if (box.get_sequence_box_count() < 2
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

        self.sequences = [s for s in self.sequences if (
            s.get_box_count() > 2
            and s.get_distance_to_center_line_avg() < 3
            and s.get_x_d_min_max_d_pct() < 16
            and s.get_h_min_max_d_pct() < 20
        )]

    def _merge_shards(self):
        shard_box_keys = []
        for box in self.boxes:
            if box.key in shard_box_keys:
                continue

            shard_box = next((b for b in self.boxes if (
                b.key != box.key
                and b.is_shard_of(box)
            )), None)
            if shard_box:
                box.merge(shard_box)
                shard_box_keys.append(shard_box.key)

        self.boxes = [b for b in self.boxes if b.key not in shard_box_keys]

    def _remove_duplicates(self):
        duplicate_box_keys = []
        for box in self.boxes:
            if box.key in duplicate_box_keys:
                continue

            for duplicate_box in (b for b in self.boxes if (
                b.key != box.key
                and b.is_duplicate_of(box)
            )):
                duplicate_box_keys.append(duplicate_box.key)

        self.boxes = [b for b in self.boxes if b.key not in duplicate_box_keys]

    def _remove_subsequences(self):
        subsequence_keys = []
        for sequence in self.sequences:
            if sequence.key in subsequence_keys:
                continue

            for subsequence in (s for s in self.sequences if (
                s.key != sequence.key
                and s.is_subsequence_of(sequence)
            )):
                subsequence_keys.append(subsequence.key)

        self.sequences = [b for b in self.sequences if b.key not in subsequence_keys]

    def get_box_count(self):
        return len(self.boxes)

    def get_sequence_count(self):
        return len(self.sequences)


class Sequence(object):

    def __init__(self, boxes):
        self.key = boxes[0].key
        self.boxes = boxes

        for box in boxes:
            box.sequence = self

        self.patched_box_count = 0

    def get_top_line(self):
        return extend_line((
            self.boxes[0].source_contour.et_point,
            self.boxes[-1].source_contour.et_point
        ), 2)

    def get_bottom_line(self):
        return extend_line((
            self.boxes[0].source_contour.eb_point,
            self.boxes[-1].source_contour.eb_point
        ), 2)

    def get_left_line(self):
        return self.boxes[0].get_left_vertical_line()

    def get_right_line(self):
        return self.boxes[-1].get_right_vertical_line()

    def get_contour(self):
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

    def get_extended_contour(self):
        contour = self.get_contour()
        i = 5

        return Tetragon(
            (contour.tl_point[0] - i, contour.tl_point[1] - i),
            (contour.tr_point[0] + i, contour.tr_point[1] - i),
            (contour.br_point[0] + i, contour.br_point[1] + i),
            (contour.bl_point[0] - i, contour.bl_point[1] + i)
        )

    def get_box_count(self):
        return len(self.boxes) + self.patched_box_count

    def patch_prepend_box(self, box, patched_box_count=1):
        box.sequence = self
        self.boxes.insert(0, box)
        self.patched_box_count += patched_box_count

    def patch_append_box(self, box, patched_box_count=1):
        box.sequence = self
        self.boxes.append(box)
        self.patched_box_count += patched_box_count

    def get_x_ds(self):
        x_ds = []

        for previous_box, box in zip(self.boxes, self.boxes[1:]):
            a_contour = previous_box.backbone_contour
            b_contour = box.backbone_contour
            x_ds.append(b_contour.c_point[0] - a_contour.c_point[0])

        return x_ds

    def get_x_d_avg(self):
        x_ds = self.get_x_ds()

        return sum(x_ds) / float(len(x_ds))

    def get_x_d_min_max_d_pct(self):
        x_ds = self.get_x_ds()

        if self.patched_box_count > 0:
            x_ds.sort()
            max_x_d = x_ds.pop(-1)
            x_ds.append(max_x_d / 2)
            x_ds.append(max_x_d / 2)

        return get_d_abs_pct(max(x_ds), min(x_ds))

    def get_hs(self):
        return [b.backbone_contour.h for b in self.boxes]

    def get_h_min_max_d_pct(self):
        hs = self.get_hs()

        return get_d_abs_pct(max(hs), min(hs))

    def get_distance_to_center_line_avg(self):
        distances = []
        for box in self.boxes[1:-1]:
            distance = get_point_to_line_distance(
                box.contour.c_point,
                (self.boxes[0].contour.c_point, self.boxes[-1].contour.c_point)
            )
            distances.append(distance)

        return sum(distances) / float(len(distances))

    def is_subsequence_of(self, sequence):
        return sequence.get_contour().contains_point(self.get_contour().c_point)


class Box(object):

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
        if tl_point is None:
            x, y, w, h = cv2.boundingRect(points)
            tl_point = x, y

        if self.source_contour is not None:
            points = np.concatenate((self.source_contour.points, points))

        self.source_contour = Polygon(points)
        self.contour = Rectangle(tl_point, w, h)
        self.key = self.contour.c_point

        if self.contour.aspect_ratio > 0.4:
            x, y = tl_point

            self.backbone_contour = Rectangle(
                (x + 2 / 3 * w, y),
                1 / 3 * w,
                h
            )
            self.tl_contour = Rectangle(
                (x, y + 1 / 7 * h),
                1 / 3 * w,
                2 / 7 * h
            )
            self.t_contour = Rectangle(
                (x + 1 / 3 * w, y),
                1 / 3 * w,
                1 / 7 * h
            )
            self.tr_contour = Rectangle(
                (x + 2 / 3 * w, y + 1 / 7 * h),
                1 / 3 * w,
                2 / 7 * h
            )
            self.c_contour = Rectangle(
                (x + 1 / 3 * w, y + 3 / 7 * h),
                1 / 3 * w,
                1 / 7 * h
            )
            self.br_contour = Rectangle(
                (x + 2 / 3 * w, y + 4 / 7 * h),
                1 / 3 * w,
                2 / 7 * h
            )
            self.b_contour = Rectangle(
                (x + 1 / 3 * w, y + 6 / 7 * h),
                1 / 3 * w,
                1 / 7 * h
            )
            self.bl_contour = Rectangle(
                (x, y + 4 / 7 * h),
                1 / 3 * w,
                2 / 7 * h
            )
        else:
            self.backbone_contour = self.contour
            self.is_tall_narrow = True

    def _process_sequence_boxes(self, box):
        if not box:
            return

        self.sequence_boxes.append(box)
        self._process_sequence_boxes(box.next)

    def is_next_after(self, box, is_patching=False):
        a_contour = box.backbone_contour
        b_contour = self.backbone_contour

        x_d = b_contour.c_point[0] - a_contour.c_point[0]
        y_d_abs = abs(b_contour.c_point[1] - a_contour.c_point[1])
        w_d_abs = abs(b_contour.w - a_contour.w)
        h_d_abs = abs(b_contour.h - a_contour.h)

        min_x_d = 0
        max_x_d = 0.95 * a_contour.h

        if is_patching:
            # Should be twice the x_d of the sequence
            sequence = self.sequence if self.sequence else box.sequence
            expected_x_d = 2 * sequence.get_x_d_avg()
            min_x_d = expected_x_d - expected_x_d * 0.1
            max_x_d = expected_x_d + expected_x_d * 0.03

        max_y_d_abs = 0.25 * a_contour.h
        max_w_d_abs = 0.50 * a_contour.w
        max_h_d_abs = 0.25 * a_contour.h

        is_next = (
            # Main
            # cv2.pointPolygonTest(box.contour.points, self.contour.c_point, False) < 0
            # and cv2.pointPolygonTest(self.contour.points, box.contour.c_point, False) < 0
            self.contour.tl_point[0] - box.contour.tr_point[0] >= 0 # Right margin
            # Backbone
            and (x_d > min_x_d and x_d < max_x_d)
            and y_d_abs < max_y_d_abs
            and w_d_abs < max_w_d_abs
            and h_d_abs < max_h_d_abs
        )

        # if self.key[0] == 262:
        #     print "(%i, %i) > (%i, %i) = %r" % (box.key[0], box.key[1], self.key[0], self.key[1], is_next)
        #     print "    x_d=%.2f [%.2f, %.2f]" % (x_d, min_x_d, max_x_d)
        #     print "    y_d_abs=%.2f [%.2f]" % (y_d_abs, max_y_d_abs)
        #     print "    w_d_abs=%.2f [%.2f]" % (w_d_abs, max_w_d_abs)

        return is_next

    def is_duplicate_of(self, box):
        a_contour = box.contour
        b_contour = self.contour

        x_d_abs = abs(b_contour.c_point[0] - a_contour.c_point[0])
        y_d_abs = abs(b_contour.c_point[1] - a_contour.c_point[1])
        w_d_abs = abs(a_contour.w - b_contour.w)
        h_d_abs = abs(a_contour.h - b_contour.h)

        return (
            x_d_abs < 6
            and y_d_abs < 6
            and w_d_abs < 6
            and h_d_abs < 6
        )

    def is_shard_of(self, box):
        a_contour = box.contour
        b_contour = self.contour

        x_d_abs = abs(b_contour.c_point[0] - a_contour.c_point[0])
        y_d = b_contour.c_point[1] - a_contour.c_point[1]
        w_d_abs = abs(a_contour.w - b_contour.w)
        h_d_abs = abs(a_contour.h - b_contour.h)

        return (
            x_d_abs < 8
            and y_d > 0
            and abs(a_contour.h - y_d) < 6
            and w_d_abs < 6
            and h_d_abs < 6
        )

    def merge(self, shard_box):
        x = min(self.contour.tl_point[0], shard_box.contour.bl_point[0])
        y = self.contour.tl_point[1]
        w = max(self.contour.tr_point[0], shard_box.contour.br_point[0]) - x
        h = shard_box.contour.bl_point[1] - self.contour.tl_point[1]

        self._set_contours(shard_box.source_contour.points, (x, y), w, h)

    def set_sequence_boxes(self):
        self._process_sequence_boxes(self)

    def get_sequence_box_count(self):
        return len(self.sequence_boxes)

    def get_left_vertical_line(self):
        line, i_point, e_point_to_line_distance = self.get_vertical_line_with_context()

        if i_point[0] < self.contour.c_point[0]:
            return line
        else:
            a_point, b_point = line

            return (
                (int(a_point[0] - e_point_to_line_distance), a_point[1]),
                (int(b_point[0] - e_point_to_line_distance), b_point[1])
            )

    def get_right_vertical_line(self):
        line, i_point, e_point_to_line_distance = self.get_vertical_line_with_context()

        if i_point[0] > self.contour.c_point[0]:
            return line
        else:
            a_point, b_point = line

            return (
                (int(a_point[0] + e_point_to_line_distance), a_point[1]),
                (int(b_point[0] + e_point_to_line_distance), b_point[1])
            )

    def get_vertical_line_with_context(self):
        line = self.get_extended_vertical_line()
        e_point = sorted(
            self.source_contour.get_points(),
            key=lambda p: get_point_to_line_distance(p, line),
            reverse=True
        )[0]

        return (
            line,
            get_intersection_point(line, self.contour.get_center_line()),
            get_point_to_line_distance(e_point, line),
        )

    def get_extended_vertical_line(self):
        a_point, b_point = self.get_vertical_line()
        y_d = b_point[1] - a_point[1]
        extend_factor = (self.contour.h / y_d) * 2

        return extend_line((a_point, b_point), extend_factor)

    def get_vertical_line(self):
        return sorted(
            self.source_contour.get_vertical_lines(),
            key=lambda l: abs(l[1][1] - l[0][1]),
            reverse=True
        )[0]

    def get_segment_points(self, segment_contour):
        points = []

        for x in range(int(segment_contour.tl_point[0]), int(segment_contour.tr_point[0])):
            for y in range(int(segment_contour.tl_point[1]), int(segment_contour.bl_point[1])):
                point = x, y

                if cv2.pointPolygonTest(self.source_contour.points, point, False) > 0:
                    points.append(point)

        return points


class Polygon(object):

    def __init__(self, points):
        self.points = points

        self.el_point = tuple(points[points[:, :, 0].argmin()][0])
        self.et_point = tuple(points[points[:, :, 1].argmin()][0])
        self.er_point = tuple(points[points[:, :, 0].argmax()][0])
        self.eb_point = tuple(points[points[:, :, 1].argmax()][0])

        self.moments = cv2.moments(points)
        self.c_point = (
            int(self.moments['m10'] / self.moments['m00']),
            int(self.moments['m01'] / self.moments['m00'])
        ) if self.moments['m00'] != 0 else None

        self.w = self.er_point[0] - self.el_point[0]
        self.h = self.eb_point[1] - self.et_point[1]
        self.aspect_ratio = self.w / float(self.h) if self.h > 0 else 0
        self.area = cv2.contourArea(points)
        self.perimeter = cv2.arcLength(points, True)

    def contains_point(self, point):
        return cv2.pointPolygonTest(self.points, point, False) > 0

    def contains_contour(self, contour):
        return (
            self.contains_point(contour.el_point)
            and self.contains_point(contour.et_point)
            and self.contains_point(contour.er_point)
            and self.contains_point(contour.eb_point)
        )

    def get_vertical_lines(self):
        points = self.get_points()
        vertical_segments = []

        for i, a_point in enumerate(points):
            if i + 2 > len(points):
                break

            if vertical_segments and a_point in vertical_segments[-1]:
                continue

            segment = [a_point]

            for b_point in points[i + 2:]:
                if ((vertical_segments and b_point in vertical_segments[-1])
                    or abs(b_point[1] - a_point[1]) < 5
                ):
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

    def get_all_points(self):
        points = self.get_points()
        all_points = []

        for a_point, b_point in zip(points, points[1:]):
            all_points.extend(get_points((a_point, b_point)))

        return all_points

    def get_points(self):
        return [tuple(p[0]) for p in self.points]


class Tetragon(Polygon):

    def __init__(self, tl_point, tr_point, br_point, bl_point):
        super(Tetragon, self).__init__(np.array([
            [tl_point],
            [tr_point],
            [br_point],
            [bl_point],
        ], dtype=np.int32))

        self.tl_point = tl_point
        self.tr_point = tr_point
        self.br_point = br_point
        self.bl_point = bl_point

    def get_center_line(self):
        return (
            (self.tl_point[0], self.c_point[1]),
            (self.tr_point[0], self.c_point[1])
        )


class Rectangle(Tetragon):

    def __init__(self, tl_point, w, h):
        x, y = tl_point

        super(Rectangle, self).__init__(
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        )

    @classmethod
    def from_tl_point_br_point(cls, tl_point, br_point):
        return cls(
            tl_point,
            br_point[0] - tl_point[0],
            br_point[1] - tl_point[1]
        )


class Point(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return 'Point(x={},y={})'.format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Line(object):

    def __init__(self, point_one, point_two):
        self.point_one = point_one
        self.point_two = point_two

    def __str__(self):
        return 'Line(p1:{},p2:{})'.format(self.point_one, self.point_two)

    @property
    def points(self):
        return self.point_one, self.point_two

    @property
    def length(self):
        return ((self.point_one.x - self.point_two.x)**2 + (self.point_one.y - self.point_two.y)**2)**0.5

    def scale(self, factor):
        self.point_one.x, self.point_two.x = Line.scale_dimension(self.point_one.x, self.point_two.x, factor)
        self.point_one.y, self.point_two.y = Line.scale_dimension(self.point_one.y, self.point_two.y, factor)

    @staticmethod
    def scale_dimension(dim1, dim2, factor):
        base_length = dim2 - dim1
        ret1 = dim1 - (base_length * (factor - 1) / 2)
        ret2 = dim2 + (base_length * (factor - 1) / 2)
        return ret1, ret2


# http://stackoverflow.com/questions/28825461/how-to-extend-a-line-segment-in-both-directions
def extend_line(line, factor):
    L = Line(Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1]))
    L.scale(factor)

    return (int(L.points[0].x), int(L.points[0].y)), (int(L.points[1].x), int(L.points[1].y))


# http://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
def _get_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def get_intersection_point(line1, line2):
    a_point, b_point = line1
    c_point, d_point = line2

    L1 = _get_line(a_point, b_point)
    L2 = _get_line(c_point, d_point)
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment
def get_point_to_line_distance(point, line):
    px, py = point
    x1, y1 = line[0]
    x2, y2 = line[1]

    line_mag = math.hypot(x2 - x1, y2 - y1)

    if line_mag < 0.00000001:
        return 9999

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (line_mag * line_mag)

    if (u < 0.00001) or (u > 1):
        ix = math.hypot(x1 - px, y1 - py)
        iy = math.hypot(x2 - px, y2 - py)
        if ix > iy:
            point_to_line_distance = iy
        else:
            point_to_line_distance = ix
    else:
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        point_to_line_distance = math.hypot(ix - px, iy - py)

    return point_to_line_distance


def get_m_point(line):
    a_point, b_point = line

    return (a_point[0] + b_point[0]) / 2, (a_point[1] + b_point[1]) / 2


# http://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python
def get_points(line):
    x1, y1 = line[0]
    x2, y2 = line[1]

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def get_d_abs_pct(a, b):
    part = abs(float(a - b))
    whole = float(a)

    return round(part / whole * 100)
