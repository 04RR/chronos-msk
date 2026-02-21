import argparse
import os
import time

# Import Agents
from agents.agent1_scout import ScoutAgent
from agents.agent2_radiologist import RadiologistAgent
from agents.agent3_archivist import ArchivistAgent
from agents.agent4_anthropologist import AnthropologistAgent
from agents.agent5_regressor import RegressorAgent

# --- PATH CONFIGURATION ---
# Update these paths to point to your real weights
SCOUT_WEIGHTS = "weights/best_scout.pt"
SVM_WEIGHTS = "weights/radiologist_head.pkl"
INDICES_DIR = "indices"
REGRESSOR_DIR = "weights/medsiglip_sota"


def main():
    parser = argparse.ArgumentParser(description="Chronos-MSK Orchestrator")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument(
        "--sex", required=True, choices=["Male", "Female"], help="Biological Sex"
    )
    parser.add_argument(
        "--race",
        required=True,
        choices=["Asian", "Black", "White", "Hispanic", "Other"],
    )
    args = parser.parse_args()

    print("\n🚀 Initializing Chronos-MSK System...")
    start_time = time.time()

    # 1. Load Agents
    try:
        scout = ScoutAgent(SCOUT_WEIGHTS)
        radiologist = RadiologistAgent(SVM_WEIGHTS)
        archivist = ArchivistAgent(INDICES_DIR)
        regressor = RegressorAgent(REGRESSOR_DIR)
        anthropologist = AnthropologistAgent()
        print("✅ All Agents Online.\n")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Pipeline Execution
    try:
        print(f"📸 Processing: {args.image}")

        # Step A: Scout (Detection & Crop)
        crop_bgr = scout.predict(args.image)

        # Step B: Radiologist (Classification & Embedding)
        stage, embedding = radiologist.predict(crop_bgr)
        print(f"   > Radiologist: Stage {stage}")

        # Step C: Archivist (Retrieval)
        matches = archivist.retrieve(embedding, args.sex, args.race)
        print(f"   > Archivist: Found {len(matches)} matches")

        # Step D: Regressor (Raw Age)
        is_male = args.sex == "Male"
        reg_age = regressor.predict(args.image, is_male)
        print(f"   > Regressor: {reg_age:.2f} years")

        # Step E: Anthropologist (Final Consensus)
        result = anthropologist.analyze(args.sex, args.race, stage, matches, reg_age)

        # 3. Final Output
        print("\n" + "=" * 30)
        print("📝 FINAL FORENSIC REPORT")
        print("=" * 30)
        print(f"Estimated Age: {result['final_age']} years")
        print(f"Maturity Stage: TW3 {result['stage']}")
        print(f"Confidence:    {result['safety_flag']}")
        print(
            f"Raw Inputs:    Reg={result['raw_regressor']}y | Ret={result['raw_archivist']}y"
        )
        print("=" * 30)

        print(f"\nTotal Time: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")


if __name__ == "__main__":
    main()
