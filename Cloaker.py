import torch
import io
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

class AegisCloak:
    def __init__(self):
        print("🛡️ Aegis Engine v3 (Hotspot Suppression) Online")
        
        # 1. Hardware Check
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print(f"💻 Hardware: {self.device.upper()} (Warning: Processing will be slower)")
        else:
            print(f"✅ Hardware: {self.device.upper()}")

        # 2. Load YOLO
        print("🚀 Loading YOLOv8n...")
        self.model = YOLO("yolov8n.pt")
        
        # 3. Preprocessing Pipeline
        self.preprocess = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    def vanish(self, image_bytes, steps, epsilon):
        try:
            # 1. Load Image
            original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = original_pil.size
            
            # Convert to Tensor
            img_tensor = self.preprocess(original_pil).unsqueeze(0).to(self.device)
            
            # 2. THE SCAN: Identify "Hotspots" (Where is the AI looking?)
            # We run inference ONCE on the clean image to find the enemy.
            print("🔍 Scanning for detection hotspots...")
            with torch.no_grad():
                clean_preds = self.model.model(img_tensor)
                if isinstance(clean_preds, (list, tuple)):
                    clean_preds = clean_preds[0]
                
                # Focus on Class Probabilities (Rows 4-84)
                clean_scores = clean_preds[:, 4:, :]
                
                # Find indices where confidence is high (> 0.4)
                # We create a "Mask" of the object.
                # We flatten to make indexing easier.
                flat_scores = clean_scores.flatten()
                
                # We take the Top 1000 most active neurons (the object)
                # This effectively "locks on" to the person/car/dog.
                _, target_indices = torch.topk(flat_scores, 1000)
            
            # 3. The Noise (Delta)
            delta = torch.zeros_like(img_tensor, requires_grad=True, device=self.device)
            
            # Momentum buffer for MI-FGSM
            momentum = torch.zeros_like(img_tensor, device=self.device)
            decay = 1.0 # Momentum decay factor
            alpha = epsilon / steps * 2 # Heuristic step size

            print(f"⚡ Laser Attack Started: {steps} steps, Eps: {epsilon}")

            for i in range(steps):
                if delta.grad is not None:
                    delta.grad.zero_()
                
                # Apply noise
                adv_img = torch.clamp(img_tensor + delta, 0, 1)
                
                # Forward Pass
                preds = self.model.model(adv_img)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                
                # 4. THE UPGRADED LOSS: "Hotspot Destruction"
                # We only minimize the scores at the INDICES we found in the scan.
                # This ignores background noise and focuses purely on erasing the object.
                current_scores = preds[:, 4:, :].flatten()
                target_scores = current_scores[target_indices]
                
                # Loss: Sum of specific hot neurons
                loss = torch.sum(target_scores)
                
                loss.backward()
                
                # 5. Update with Momentum (MI-FGSM)
                # This helps punch through local minima
                if delta.grad is not None:
                    grad = delta.grad.detach()
                    
                    # Normalize gradient (L1 norm)
                    grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                    
                    # Update momentum
                    momentum = decay * momentum + grad
                    
                    # Update delta
                    delta.data = delta.data - alpha * torch.sign(momentum)
                    
                    # Clamp to Epsilon (Invisibility Shield)
                    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                    delta.data = torch.clamp(img_tensor + delta.data, 0, 1) - img_tensor
                
                if i % 10 == 0:
                    print(f"Step {i}: Hotspot Energy = {loss.item():.4f}")

            # 6. Finalize
            cloaked_tensor = torch.clamp(img_tensor + delta, 0, 1)
            final_img = T.ToPILImage()(cloaked_tensor.squeeze(0).cpu())
            
            # High-Quality Resize
            final_img = final_img.resize(original_size, Image.LANCZOS)
            
            return final_img
            
        except Exception as e:
            print(f"Error in backend: {e}")
            raise e