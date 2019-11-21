# -*- coding: utf-8 -*-
import cv2
import os.path
import datetime


import images
import contours
import colors


class NumericVision(object):

    @staticmethod
    def process(image_path, roi_contour=None):
        assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path

        out_path = "images/out/%s/" % '{0:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        original_image = cv2.imread(image_path)
        cv2.imwrite(out_path + '1.jpg', original_image)
        filtered_image = images.filter(original_image)
        cv2.imwrite(out_path + '2.jpg', filtered_image)

        image_contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bag = contours.Bag(filtered_image, image_contours, hierarchy[0], roi_contour)

        print("%i/%i" % (len(image_contours), bag.get_box_count()))

        contours_image = cv2.imread(image_path)
        if roi_contour:
            cv2.drawContours(contours_image, [roi_contour.points], -1, colors.BLUE, 3)

        color = colors.GREEN
        thickness = 1
        for sequence in bag.sequences:
            cv2.drawContours(contours_image, [sequence.get_contour().points], -1, color, thickness)
            cv2.line(contours_image, sequence.boxes[0].contour.c_point, sequence.boxes[-1].contour.c_point, color, thickness)
            cv2.putText(contours_image, "(%i,%i)" % sequence.key, sequence.boxes[-1].contour.c_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            print("%i(%i/%.2f)" % (
                sequence.key[0],
                sequence.get_box_count(),
                sequence.get_h_min_max_d_pct(),
            ))

            for box in sequence.boxes:
                # cv2.drawContours(contours_image, [box.contour.points], -1, colors.RED, thickness)
                cv2.circle(contours_image, box.contour.c_point, 1, color, thickness)

            contour = sequence.get_contour()
            transformed_image = images.four_point_transform(
                original_image,
                contour.tl_point,
                contour.tr_point,
                contour.br_point,
                contour.bl_point
            )
            cv2.imwrite(out_path + "seq_%i_%i.jpg" % sequence.key, transformed_image)

        cv2.imwrite(out_path + '3.jpg', contours_image)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    # http://nicodjimenez.github.io/boxLabel/annotate.html
    # roi_contour = contours.Rectangle.from_tl_point_br_point((1271, 2376), (1393, 2459))

    NumericVision.process(args[0]) #, roi_contour)
