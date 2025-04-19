from ultralytics.utils import ASSETS, LOGGER
import torch
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.torch_utils import select_device
from ultralytics import YOLO

class AutoAnnotator:
    def __init__(self, model=None, device="", variant="base"):
        self.YOLO_MODEL = True  # Assume YOLO by default

        if model is None:
            LOGGER.warning(f"⚠️ No model provided. Using default: yolo11n.pt")
            self.model = "yolo11n.pt"
        else:
            model_str = str(model).casefold()

            if model_str.endswith(".pt") and model_str != "florence2.pt":
                self.model = YOLO(model)
            elif model_str in ("florence2.pt", "florence2"):
                self.model = Florence2(device=device, variant=variant)
                self.YOLO_MODEL = False
            else:
                LOGGER.warning(
                    f"⚠️ Unsupported model '{model}'. Using default: yolo11n.pt.\n"
                    f"✅ Supported models:\n"
                    f"   - YOLO: yolov8n.pt, yolov11n.pt, yolov10.pt, yolo11n.pt, yolo12n.pt\n"
                    f"   - Florence2: florence2"
                )
                self.model = YOLO("yolo11n.pt")

        if self.YOLO_MODEL:
            self.names = self.model.names

    @staticmethod
    def convert_to_yolo(x1, y1, x2, y2, w, h):
        """Convert bounding box coordinates from (x1, y1, x2, y2) to normalized YOLO format."""
        cx, cy, bw, bh = xyxy2xywhn(torch.tensor([[x1, y1, x2, y2]]), w=w, h=h, clip=True)[0].tolist()
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def create_classes_txt(output_dir, label_map):
        """Create classes.txt file for verification of labels and advance operations"""
        try:
            with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
                    f.write(f"{label}\n")
            LOGGER.info(f"✅ Saved class names to {output_dir / 'classes.txt'}")
        except Exception as e:
            LOGGER.error(f"❌ Failed to save classes.txt: {e}")

    def annotate(
        self,
        source=None,
        conf=0.25,  # not supported for florence2
        iou=0.45,   # not supported for florence2
        classes=None,
        save=True,
        output_dir="labels",
        save_visuals=True,
        visuals_output_dir=None
    ):
        import time
        from pathlib import Path
        from tqdm import tqdm

        if save_visuals:
            import cv2

        start = time.time()

        if source is None:
            LOGGER.warning("⚠️ 'source' argument is missing. Using default source.")
            source = ASSETS

        dataset = LoadImagesAndVideos(path=str(source))
        if not dataset:
            LOGGER.warning("⚠️ No images or videos found in the source.")
            return

        output_dir = Path(output_dir)
        visuals_output_dir = Path(visuals_output_dir or output_dir.parent / f"{output_dir.name}_visuals")

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        if save_visuals:
            visuals_output_dir.mkdir(parents=True, exist_ok=True)

        label_map = {}
        processed = 0
        is_yolo = self.YOLO_MODEL

        # Convert user defined names to YOLO classes for processing
        if is_yolo:
            classes = [i for i, name in self.names.items() if name in classes]

        for path, im0, _ in tqdm(dataset, desc="Annotating Images"):
            try:
                img_path = path[0] if isinstance(path, (list, tuple)) else path
                img_name = Path(img_path).stem
                im0 = im0[0] if isinstance(im0, (list, tuple)) else im0

                if im0 is None:
                    LOGGER.warning(f"⚠️ Could not read image: {img_path}")
                    continue

                h, w = im0.shape[:2]
                if is_yolo:
                    result = self.model.predict(im0, classes=classes, verbose=False, conf=conf, iou=iou)[0]
                    bboxes = result.boxes.xyxy.tolist()
                    labels = result.boxes.cls.tolist()
                else:
                    result = self.model.process(im0, w, h)
                    bboxes = result.get("bboxes") or []
                    labels = result.get("labels") or []

                if not bboxes or not labels:
                    LOGGER.warning(f"⚠️ No output for image: {img_name}")
                    continue

                lines = []
                if save_visuals:
                    annotator = Annotator(im0)

                found_labels = set()

                for box, label in zip(bboxes, labels):
                    if classes is None or label in classes:
                        found_labels.add(label)
                        label_name = self.names[label] if is_yolo else label
                        if label_name not in label_map:
                            label_map[label_name] = len(label_map)
                        mapped_id = label_map[label_name]

                        if save_visuals:
                            annotator.box_label(box, label=label_name, color=colors(mapped_id, True))

                        x1, y1, x2, y2 = box
                        lines.append(f"{mapped_id} {self.convert_to_yolo(x1, y1, x2, y2, w, h)}")

                if not lines:
                    if classes:
                        missing = set(classes) - found_labels
                        msg = f"Missing classes: {sorted(missing)}" if missing else "All boxes skipped."
                        LOGGER.warning(f"⚠️ Skipping image {img_name}. {msg}")
                    else:
                        LOGGER.warning(f"⚠️ All boxes skipped for: {img_name}")
                    continue

                if save:
                    (output_dir / f"{img_name}.txt").write_text("\n".join(lines), encoding="utf-8")

                if save_visuals:
                    cv2.imwrite(str(visuals_output_dir / f"{img_name}.png"), annotator.result())

                processed += 1

            except Exception as e:
                LOGGER.error(f"❌ Error processing {path}: {e}", exc_info=True)

        if save and label_map:
            self.create_classes_txt(output_dir, label_map)

        if processed:
            LOGGER.info(f"✅ Annotated {processed} image(s) in {time.time() - start:.2f} seconds.")
        else:
            LOGGER.warning("⚠️ No images were successfully annotated.")


class Florence2:
    def __init__(self, device="", variant="base"):
        self.device = select_device(device)

        supported_variants = {"base", "large", "large-ft", "base-ft"}
        self.variant = variant if variant in supported_variants else "base"

        if variant not in supported_variants:
            LOGGER.warning(f"⚠️ Invalid variant '{variant}' provided. Falling back to 'base'. "
                           f"Supported variants: {supported_variants}")

        self.mid = f"microsoft/florence-2-{self.variant}"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Florence-2 grounding model and corresponding processor."""
        from ultralytics.utils.checks import check_requirements

        check_requirements(["transformers==4.49.0", "einops"])  # Ensure required libraries

        from transformers import AutoModelForCausalLM, AutoProcessor, logging
        LOGGER.info(f"💡 Initializing Florence2-base on {str(self.device).upper()}")
        logging.set_verbosity_error()  # Suppress excessive logs from transformers library: https://huggingface.co/docs/transformers/en/main_classes/logging

        self.model = AutoModelForCausalLM.from_pretrained(self.mid, trust_remote_code=True,).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.mid, trust_remote_code=True)

        LOGGER.info(f"🚀 Loaded Florence2-{self.variant} model successfully.")

    def process(self, im0, w, h):

        # Encode image and text prompt
        inputs = self.processor(text="<OD>", images=im0, return_tensors="pt").to(self.device)
        ids, pval = inputs["input_ids"], inputs["pixel_values"]

        # Perform inference
        outids = self.model.generate(ids, pixel_values=pval, max_new_tokens=1024, early_stopping=False, num_beams=3)

        # Decode and post-process output
        outputs = self.processor.batch_decode(outids, skip_special_tokens=False)[0]
        processed = self.processor.post_process_generation(outputs, task="<OD>", image_size=(w, h))

        return processed.get("<OD>", {})  # Return decoded results for post-processing in Ultralytics YOLO format
