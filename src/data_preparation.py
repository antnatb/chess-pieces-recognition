import cv2
import os
import numpy as np
from detection import detect_black_pieces, detect_white_pieces, crop_pieces

LABELS = {
    'p': 'pawn',
    't': 'rook',
    'n': 'knight',
    'b': 'bishop',
    'q': 'queen',
    'k': 'king',
    'x': 'discard',
}

def resize_for_display(img):
    h, w = img.shape[:2]
    screen_w = 1920  # default fallback
    screen_h = 1080  # default fallback

    # Try to get screen resolution from a dummy fullscreen window
    try:
        cv2.namedWindow("_temp", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("_temp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_w = cv2.getWindowImageRect("_temp")[2]
        screen_h = cv2.getWindowImageRect("_temp")[3]
        cv2.destroyWindow("_temp")
    except:
        pass

    target_w = screen_w // 2
    target_h = screen_h // 2

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def manual_label_with_recrop(img, color, pieces):  # pieces is now list of (piece, image_name, idx)
    # Label directory
    save_path = os.path.join("data", "detected", color)
    os.makedirs(save_path, exist_ok=True)

    for piece, image_name, idx in pieces:
        if piece.shape[0] == 0 or piece.shape[1] == 0:
            print(f"Warning: Skipped empty crop for {color} piece {idx}")
            continue
        display_img = resize_for_display(piece)
        cv2.namedWindow("Auto-cropped piece", cv2.WINDOW_NORMAL)
        screen_w = 1920  # fallback width
        screen_h = 1080  # fallback height
        win_w, win_h = display_img.shape[1], display_img.shape[0]
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        # Attempt to center window with fallback coordinates
        try:
            cv2.moveWindow("Auto-cropped piece", x, y)
        except:
            pass
        cv2.imshow("Auto-cropped piece", display_img)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC to quit
            break

        if chr(key) == 'r':  # recrop
            cv2.destroyWindow("Auto-cropped piece")
            cv2.namedWindow("Recrop", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Recrop", 800, 800)
            roi = cv2.selectROI("Recrop", piece, fromCenter=False, showCrosshair=True)
            if roi[2] > 0 and roi[3] > 0:
                rx, ry, rw, rh = roi
                piece = piece[ry:ry+rh, rx:rx+rw]
                # Show recropped image for labeling
                cv2.destroyWindow("Recrop")
                display_img = piece
                display_img = resize_for_display(piece)
                cv2.namedWindow("Recropped piece", cv2.WINDOW_NORMAL)
                cv2.imshow("Recropped piece", display_img)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("Recropped piece")
                key = cv2.waitKey(0) & 0xFF

        elif chr(key) in LABELS:
            label = LABELS[chr(key)]
            if label == "discard":
                continue
            label_folder = os.path.join(save_path, label)
            os.makedirs(label_folder, exist_ok=True)
            filename = os.path.join(label_folder, f"{color}_{label}_{image_name}_{idx}.png")
            cv2.imwrite(filename, piece)

    cv2.destroyAllWindows()


def process_image(img, image_name="image"):
    # Resize and crop image similar to detect_pieces
    img_resized = cv2.resize(img, (640, 480))
    img_cropped = img_resized[10:-10, 5:-20]

    # Detect and crop black pieces
    black_contours = detect_black_pieces(img_cropped)
    black_pieces = crop_pieces(img_cropped, black_contours, "black")
    manual_label_with_recrop(img_cropped, "black", [(p, image_name, i) for i, p in enumerate(black_pieces)])

    # Detect and crop white pieces
    white_contours = detect_white_pieces(img_cropped)
    white_pieces = crop_pieces(img_cropped, white_contours, "white")
    manual_label_with_recrop(img_cropped, "white", [(p, image_name, i) for i, p in enumerate(white_pieces)])


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a single board image")
    parser.add_argument("--folder", help="Path to a folder of board images")
    args = parser.parse_args()

    if args.folder:
        image_files = glob.glob(os.path.join(args.folder, "*.jpg")) + \
                      glob.glob(os.path.join(args.folder, "*.png"))
        print(f"Found {len(image_files)} images in folder '{args.folder}'")
        for image_path in image_files:
            print(f"Processing {image_path}...")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image {image_path}.")
                continue
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            process_image(img, image_name)
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("Failed to load image.")
        else:
            image_name = os.path.splitext(os.path.basename(args.image))[0]
            process_image(img, image_name)
    else:
        print("Please provide either --image or --folder.")
