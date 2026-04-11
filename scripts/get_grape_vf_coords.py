import cv2

from glaucoma_vf.utils import get_git_root

"""
This script was used to manually generate the coordinates of the 
individual VF points from the Octopus G1 program.

The points were manually selected (see assets/grape_vf_report_coords.png)
by clicking on them with the mouse, then cv2 printed the coords of 
each clicked point to the terminal. The point coords were then recorded
for later use.
"""
REPO_ROOT = get_git_root(__file__)
ASSETS_DIR = REPO_ROOT / "assets"


def main():
    coords = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Point {len(coords)+1}: {x}, {y}")
            coords.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # type: ignore
            cv2.imshow("image", img)  # type: ignore

    img = cv2.imread(str(ASSETS_DIR / "grape_vf_report_raw.png"))
    cv2.imshow("image", img)  # type: ignore
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
