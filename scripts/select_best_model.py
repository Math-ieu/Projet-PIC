import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    results_path = "training_results.json"
    if not os.path.exists(results_path):
        logger.error("No training results found.")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    best_model = None
    best_score = -1
    
    for model_name, metrics in results.items():
        if metrics.get('status') == 'failed':
            continue
            
        # Composite score logic
        # For now, just using accuracy or a mock score
        score = metrics.get('accuracy', 0)
        
        logger.info(f"Model: {model_name}, Score: {score}")
        
        if score > best_score:
            best_score = score
            best_model = model_name
            
    if best_model:
        logger.info(f"Best model selected: {best_model} with score {best_score}")
        
        # Save metadata
        metadata = {
            "best_model": best_model,
            "score": best_score,
            "path": f"checkpoints/{best_model}_final.h5"
        }
        with open("best_model_info.json", "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        logger.warning("No valid models found.")

if __name__ == "__main__":
    main()
