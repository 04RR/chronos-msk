from ultralytics import YOLO


def main():
    # Load the Nano model (smallest, fastest)
    model = YOLO("ckpts/yolo11n.pt")

    # Train
    results = model.train(
        data="datasets/scout_dataset/scout_data.yaml",
        epochs=30,  # 30 epochs is plenty for this easy task
        imgsz=640,  # Standard resolution
        batch=16,  # Adjust if you run out of GPU memory
        name="scout_agent_v1",  # Name of the run
        device=0,  # Use GPU 0
    )

    print("Training Complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
